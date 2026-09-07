# Copyright 2026 Zachary Olkin. All rights reserved.

"""Does a predicted plan's geometry agree with the terrain it will be executed on?

Two numbers per plan, both purely geometric and neither needing a rollout:

* **contact consistency** -- for every node where the plan's own kinematics say a foot is
  planted, how close is that foot to the ground it is standing on?  A stance foot 15 cm above
  a tread is a plan that has read the terrain wrong.
* **penetration** -- how far does any collider reach *inside* the terrain, at nodes where it
  is not supposed to be touching?  A shin through a riser is the signature of a plan that
  chose to jump onto a box that is really a staircase.

That pairing is the point: the failure being targeted is MODE selection, and a wrong mode is
visible in the plan's geometry long before it is visible in whether the robot fell.  It also
means the reward is a function of ``(plan, terrain)`` alone -- so it can be computed offline
on plans already recorded, which is how it gets verified before anything is trained on it.

Three decisions here that are not free choices:

**The contact flag comes from velocity, never proximity** (see :mod:`motion_gen.plan_geometry`).
Defining contact by nearness to the ground and then rewarding nearness would be vacuous.

**The buffer nodes around a stance are exempt from the penetration penalty.**  The velocity
detector is ~50% recall by design, so the nodes immediately before and after a detected
stance are where an undetected touchdown or toe-off actually lives -- and a foot genuinely in
contact there sits at ``sd ~ 0``, which the penalty would charge as if the plan had driven it
into the ground.  Only that foot is exempt: nothing else on the robot is ever supposed to
touch, so knees, torso, forearms, hands and backpack keep the full penalty at every node.

**Penetration is an area-weighted SUM, not a mean.**  A mean over every collider point
divides the signal by the grid: most of the robot is nowhere near the ground (the torso sits
0.79 m up), so the median came out identically zero while a quarter of real plans had a foot
more than 3 cm inside the terrain.  With surface samples carrying an area weight, the sum is
the integral of penetration depth over the body's surface -- a volume, in m^3 -- which grows
with how much of the foot is buried and is smooth in a way a ``max`` is not.  All three
reductions are returned so they can be compared on the same plans rather than argued about.

**The contact kernel is therefore TWO-sided.**  Because a stance node is exempt from the
penetration penalty, a plan that buries its stance foot 10 cm into a tread would otherwise be
charged by neither term.  So the contact term grades the SIGNED clearance of the closest sole
corner: hovering and burying both cost it.  ``hover`` and ``bury`` come back separately in the
diagnostics so which one dominates is a measurement rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .plan_geometry import CONTACT_DILATE_NODES, CONTACT_SPEED_THRESH, PlanColliderModel
from .terrain.box_sdf import BoxUnionTerrain


#: Penetration weight for a body the table below does not name.
#:
#: **10x the feet.**  A foot slightly inside the terrain is what a stance foot looks like at
#: the resolution the plan is accurate to -- the measured node-0 placement residual is 1.6 mm
#: median, and a real stance sits at ``sd ~ 0`` by definition.  A SHIN inside a riser is not
#: that: nothing but a foot is supposed to touch, so any depth on another link is a statement
#: that the plan has driven the robot through the terrain.
#:
#: The measured split is what makes this worth doing: of 20k recorded plans, 24.85% put a FOOT
#: at least 3 cm inside and only 0.24% put a knee in without a foot already being in.  Under a
#: uniform weight the benign case therefore sets the scale of the whole term.
DEFAULT_PENETRATION_WEIGHT: float = 10.0

#: Per-body penetration weights, for the bodies that deviate from
#: :data:`DEFAULT_PENETRATION_WEIGHT`.
#:
#: Only the feet, and only because they are the one part of the robot the terrain is SUPPOSED
#: to touch.  Everything else -- shanks, thighs, torso, forearms, hands, backpack -- keeps the
#: default.
FOOT_PENETRATION_WEIGHT: float = 1.0
DEFAULT_BODY_PENETRATION_WEIGHTS: dict[str, float] = {
    "left_ankle_roll_link": FOOT_PENETRATION_WEIGHT,
    "right_ankle_roll_link": FOOT_PENETRATION_WEIGHT,
}


@dataclass(frozen=True)
class TerrainConsistencySpec:
    """Every constant the terrain-consistency reward reads, in one declared place.

    Declared rather than fitted, for the same reason the exteroception normalizer is: a
    threshold that is a property of some recording stops meaning the same thing between runs.
    """

    contact_speed_thresh: float = CONTACT_SPEED_THRESH
    """Sole-corner speed below which a node is stance (m/s)."""

    contact_dilate_nodes: int = CONTACT_DILATE_NODES
    """Nodes each side of a stance exempted from the penetration penalty."""

    penetration_tol_m: float = 0.005
    """Depth inside the terrain that is not charged (m).

    5 mm absorbs the FK/rot6d round-trip and the fact that a plan's feet are *meant* to reach
    the surface; it is far below the centimetres a mode error produces.
    """

    contact_deadband_m: float = 0.01
    """Half-width of the contact deadband (m) -- the band where a stance is PERFECT.

    PARC's contact loss (their Eq 8) is ``c_b * min_p |sd(p)|``, which is zero only at exact
    tangency.  This widens that to a band: a stance foot whose closest point sits anywhere
    from 1 cm above the terrain to 1 cm below it costs nothing.

    1 cm is a little over half the foot box's own thickness (18.5 mm) and about twice the
    5 mm penetration tolerance, so the two terms do not fight over the same millimetres.  It
    also absorbs what the plan is not accurate enough to resolve: the measured node-0
    placement residual of this pipeline is -1.6 mm median with 87% inside 5 cm.
    """

    hover_sigma_m: float = 0.03
    """Width of the legacy exponential contact KERNEL (m).

    Superseded by :attr:`contact_deadband_m` and the PARC-style penalty, and kept only so
    ``contact_reward`` stays comparable with everything measured before the switch (notably
    the approach-distance result, 0.859 -> 0.940, which was measured on it).
    """

    clearance_margin_m: float = 0.02
    """Clearance a NON-contact collider is asked to keep (m), for the weight-0 variant.

    Shipped unused: a swing foot legitimately skims a tread by a centimetre or two, so this
    prices normal stair motion. It is computed and returned so its magnitude is a number
    before anyone decides whether to pay it.
    """

    body_penetration_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_BODY_PENETRATION_WEIGHTS))
    """Per-body multiplier on the penetration depth, by MJCF body name.

    A body absent from this mapping takes :attr:`penetration_default_weight`.  An unknown
    NAME is refused rather than ignored: a typo would silently leave that body at the default
    and the reward would look like it had been re-weighted when it had not.
    """

    penetration_default_weight: float = DEFAULT_PENETRATION_WEIGHT
    """Multiplier for any body :attr:`body_penetration_weights` does not name."""

    contact_weight: float = 68.0
    """Reference weight on ``contact_penalty`` for the combined total.

    **Declared from the measured scales, not tuned.**  Over 6000 recorded plans the two terms'
    95th percentiles are ``contact_penalty`` 0.0147 m and ``penetration_sum_m`` 9.36 m -- a
    factor of 640 -- so equal weights would make the total a report on penetration alone.
    These two put each term at about 1.0 at its own p95, which is the only defensible starting
    point before anything has been trained.

    A task's ``RewTerm`` weight is the authority at training time; this exists so the offline
    tools and the viewer agree on ONE reference total instead of each inventing a balance.
    """

    penetration_weight: float = 0.107
    """Reference weight on ``penetration_sum_m``.  See :attr:`contact_weight`.

    **Re-derived after the per-body weights landed.**  Weighting every non-foot link 10x moved
    this term's p95 from 3.24 m to 9.36 m while leaving its MEDIAN alone (0.0089 -> 0.0087) --
    because the typical penetration is a benign stance foot, which keeps weight 1.0, and the
    tail is the wrong-mode plans that put a shin through a riser.  So the per-body weights
    changed the term's SHAPE, which is the point, and this rescales the overall MAGNITUDE back
    so the two terms stay commensurate.  It went 0.31 -> 0.107.
    """

    served_nodes: int = 12
    """Leading nodes actually executed before the next re-plan.

    Every term is also reduced over just these, because the plan's tail is where the MODE
    lives while its head is what the robot does. Reporting both keeps that a choice.
    """

    def __post_init__(self) -> None:
        """Refuse a spec that cannot produce a meaningful number."""
        if self.hover_sigma_m <= 0.0:
            raise ValueError(f"hover_sigma_m must be positive; got {self.hover_sigma_m}")
        if self.contact_speed_thresh <= 0.0:
            raise ValueError(
                f"contact_speed_thresh must be positive; got {self.contact_speed_thresh}")
        if self.served_nodes < 1:
            raise ValueError(f"served_nodes must be >= 1; got {self.served_nodes}")
        if any(w < 0.0 for w in self.body_penetration_weights.values()):
            raise ValueError(
                "a negative per-body penetration weight would turn a penalty into a bonus "
                f"for that body: {self.body_penetration_weights}")
        if self.penetration_default_weight < 0.0:
            raise ValueError(
                f"penetration_default_weight must be >= 0; got "
                f"{self.penetration_default_weight}")
        if self.contact_weight < 0.0 or self.penetration_weight < 0.0:
            raise ValueError(
                "the reference weights are magnitudes for a PENALTY total and must be "
                f">= 0; got {self.contact_weight} / {self.penetration_weight}")


def resolve_point_weights(
    model: PlanColliderModel,
    spec: TerrainConsistencySpec,
) -> torch.Tensor:
    """Per-collider-POINT penetration weight, from the per-body table.

    Resolved by body NAME and refused on an unknown one, because the failure mode of a silent
    miss is a reward that reads as re-weighted while the body in question is still at the
    default.

    Args:
        model: The collider model, which supplies the body order and each point's body.
        spec: The declared weights.

    Returns:
        ``(P,)`` weight per collider point, on ``model``'s device and dtype.

    Raises:
        ValueError: If ``spec.body_penetration_weights`` names a body the model does not have.
    """
    unknown = set(spec.body_penetration_weights) - set(model.body_order)
    if unknown:
        raise ValueError(
            f"body_penetration_weights names {sorted(unknown)}, which this collider model "
            f"does not have. Its bodies are {model.body_order}. Left unchecked those entries "
            "would do nothing and the reward would look re-weighted when it is not."
        )
    per_body = torch.tensor(
        [spec.body_penetration_weights.get(b, spec.penetration_default_weight)
         for b in model.body_order],
        dtype=model.points.dtype, device=model.points.device)
    return per_body[model.body_index]


def _masked_mean(value: torch.Tensor, keep: torch.Tensor, dims: tuple[int, ...]
                 ) -> torch.Tensor:
    """Mean of ``value`` over ``dims``, counting only where ``keep``; 0 where nothing is kept.

    Args:
        value: The quantity to average.
        keep: Bool mask broadcastable to ``value``.
        dims: Dimensions to reduce.

    Returns:
        The reduced mean, with 0 wherever the mask kept nothing.
    """
    w = keep.to(value.dtype)
    total = (value * w).sum(dim=dims)
    count = w.sum(dim=dims)
    return torch.where(count > 0, total / count.clamp_min(1.0), torch.zeros_like(total))


def _pen_reductions(
    depth: torch.Tensor,
    charged: torch.Tensor,
    area: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Every penetration reduction of one node chunk, to be accumulated over chunks.

    Args:
        depth: ``(N, f, P)`` penetration depth (m), already clamped at 0.
        charged: ``(N, f, P)`` bool, ``False`` where a point is exempt.
        area: ``(P,)`` surface area each point represents (m^2).
        weight: ``(P,)`` per-body penetration multiplier (see :func:`resolve_point_weights`).

    Returns:
        Mapping with the chunk sums ``sum_m`` (PARC Eq 7, per-body weighted), ``volume``
        (area-weighted),
        ``sq`` (area-weighted, squared), ``mean_num`` / ``mean_den`` (unweighted mean of
        squares), and the per-node ``node_sum_m`` / ``node_volume``.
    """
    keep = charged.to(depth.dtype)
    kw = keep * weight
    w = keep * area * weight
    node_sum = (depth * kw).sum(dim=2)                      # (N, f) metres, PARC Eq 7
    node_vol = (depth * w).sum(dim=2)                       # (N, f) m^3
    return {
        "sum_m": node_sum.sum(dim=1),
        "volume": node_vol.sum(dim=1),
        "sq": (depth.pow(2) * w).sum(dim=(1, 2)),
        "mean_num": (depth.pow(2) * kw).sum(dim=(1, 2)),
        "mean_den": keep.sum(dim=(1, 2)),
        "node_sum_m": node_sum,
        "node_volume": node_vol,
    }


def terrain_consistency(
    points: torch.Tensor,
    contact: dict[str, torch.Tensor],
    terrain: BoxUnionTerrain,
    model: PlanColliderModel,
    spec: TerrainConsistencySpec = TerrainConsistencySpec(),
    chunk_nodes: int | None = None,
) -> dict[str, torch.Tensor]:
    """Grade a batch of plans against a batch of terrains.

    Args:
        points: ``(N, F, P, 3)`` world-frame collider points, from
            :func:`~motion_gen.plan_geometry.plan_collider_points`.
        contact: The mapping :func:`~motion_gen.plan_geometry.plan_contact` returns.
        terrain: ``N`` box-union terrains in the SAME world frame as ``points``.
        model: The collider model ``points`` was built with.
        spec: The declared constants.
        chunk_nodes: Nodes per SDF pass, or ``None`` for all at once.  The SDF's working set
            is ``N * nodes * P * B``, which at 512 envs, 64 nodes, 1886 surface points and 3
            boxes is 1.4 GB -- so the reward chunks and accumulates rather than materialising
            it.  Every per-node output is small and is kept in full.

    Returns:
        A mapping of ``(N,)`` scalars and per-node diagnostics.

        **The two reward terms, both PENALTIES -- wire them with NEGATIVE weights:**

        * ``contact_penalty`` -- PARC Eq 8 widened by :attr:`contact_deadband_m`, averaged
          over nodes: ``mean_t sum_b c_b * max(0, min_{p in sole(b)} |sd(p)| - deadband)``.
        * ``penetration_sum_m`` -- PARC Eq 7, averaged over nodes:
          ``mean_t sum_i max(0, -sd(p_i) - tol)`` over non-exempt points.

        Alternative reductions of the same geometry, so they stay comparable:
        ``penetration_volume_m3`` (area-weighted, and the DENSITY-INVARIANT twin of
        ``penetration_sum_m``), ``penetration_sq``, ``penetration_mean``,
        ``max_penetration_m``, and the legacy ``contact_reward`` kernel.  **The
        area-weighted keys are ABSENT on an extreme-point model**, which has no areas; a
        zero there would read as "nothing penetrates".

        The combined reference total, using :attr:`TerrainConsistencySpec.contact_weight`
        and :attr:`~TerrainConsistencySpec.penetration_weight`: ``weighted_contact``,
        ``weighted_penetration``, ``total_penalty``, and the per-node
        ``node_weighted_contact`` / ``node_weighted_penetration`` / ``node_total_penalty``.
        These are PENALTIES; a reward is ``-total_penalty``.

        Per-body attribution: ``penetration_by_body`` ``(N, n_bodies)``, the weighted depth
        sum charged to each body, in ``model.body_order`` -- the columns are named by the
        MODEL rather than by an entry in this mapping, which is otherwise all tensors.  This
        is what the per-body weights exist to be read against.

        Bookkeeping: ``clearance``, ``support_fraction``, ``contact_coverage``,
        ``exempt_fraction``.  Per-node: ``node_contact_penalty`` / ``node_penetration_m``
        ``(N, F)``, ``node_sd_min`` ``(N, F, n_bodies)``, ``node_foot_clearance`` (most
        penetrating sole point) / ``node_foot_abs`` (CLOSEST sole point, what Eq 8 reads) /
        ``node_hover`` / ``node_bury`` ``(N, F, 2)``.  Every scalar again with a ``_served``
        suffix over the leading ``spec.served_nodes`` nodes.

    Raises:
        ValueError: If the batch sizes of ``points`` and ``terrain`` disagree.
    """
    n, f, p = points.shape[:3]
    if len(terrain) != n:
        raise ValueError(
            f"terrain holds {len(terrain)} tiles but points has batch {n}; a broadcast here "
            "would grade every plan against the wrong tile and still return a plausible "
            "number.")
    dev, dt = points.device, points.dtype
    nb = len(model.body_order)
    area = model.areas.to(dev, dt)
    radii = model.radii.to(dev, dt)
    pt_weight = resolve_point_weights(model, spec).to(dev, dt)
    buffered = contact["contact_buffered"]                        # (N, F, 2)
    strict = contact["contact"]                                   # (N, F, 2)
    sole = model.sole_index                                        # (2, n_sole)
    k = min(int(spec.served_nodes), f)

    foot_own = torch.stack([model.body_index == b for b in model.foot_index.tolist()])

    acc = {key: torch.zeros(n, dtype=dt, device=dev) for key in
           ("sum_m", "volume", "sq", "mean_num", "mean_den", "exempt",
            "sum_m_served", "volume_served")}
    max_pen = torch.zeros(n, dtype=dt, device=dev)
    node_sum_m = torch.zeros(n, f, dtype=dt, device=dev)
    node_vol = torch.zeros(n, f, dtype=dt, device=dev)
    node_sd_min = torch.empty(n, f, nb, dtype=dt, device=dev)
    foot_clear = torch.empty(n, f, 2, dtype=dt, device=dev)
    foot_abs = torch.empty(n, f, 2, dtype=dt, device=dev)
    # Weighted depth summed per BODY: which link is being charged, which is the question the
    # per-body weights exist to answer and which the scalar cannot.
    by_body = torch.zeros(n, nb, dtype=dt, device=dev)

    step = int(chunk_nodes) if chunk_nodes else f
    for lo in range(0, f, step):
        hi = min(lo + step, f)
        c = hi - lo
        sd = terrain.sdf(points[:, lo:hi].reshape(n, c * p, 3)).reshape(n, c, p)
        sd_eff = sd - radii
        ex = torch.zeros(n, c, p, dtype=torch.bool, device=dev)
        for j_foot in range(len(model.foot_index)):
            ex |= buffered[:, lo:hi, j_foot : j_foot + 1] & foot_own[j_foot][None, None, :]
        charged = ~ex
        depth = (-sd_eff - spec.penetration_tol_m).clamp_min(0.0)

        red = _pen_reductions(depth, charged, area, pt_weight)
        by_body.scatter_add_(
            1, model.body_index[None, :].expand(n, p),
            (depth * charged.to(dt) * pt_weight).sum(dim=1))
        for key in ("sum_m", "volume", "sq", "mean_num", "mean_den"):
            acc[key] += red[key]
        node_sum_m[:, lo:hi] = red["node_sum_m"]
        node_vol[:, lo:hi] = red["node_volume"]
        acc["exempt"] += ex.to(dt).sum(dim=(1, 2))
        max_pen = torch.maximum(
            max_pen, torch.where(charged, depth, torch.zeros_like(depth)).amax(dim=(1, 2)))
        if lo < k:
            end = min(hi, k) - lo
            red_s = _pen_reductions(depth[:, :end], charged[:, :end], area, pt_weight)
            acc["sum_m_served"] += red_s["sum_m"]
            acc["volume_served"] += red_s["volume"]

        node_sd_min[:, lo:hi] = sd_eff.new_full((n, c, nb), float("inf")).scatter_reduce(
            2, model.body_index[None, None, :].expand(n, c, p), sd_eff, reduce="amin")
        per_sole = sd_eff[:, :, sole.reshape(-1)].reshape(n, c, 2, -1)
        foot_clear[:, lo:hi] = per_sole.amin(dim=-1)               # most-penetrating point
        foot_abs[:, lo:hi] = per_sole.abs().amin(dim=-1)           # CLOSEST point, PARC Eq 8

    # ---- PARC Eq 8, widened by the deadband -------------------------------------------
    # ``min_p |sd|``, not ``|min_p sd|``: the closest point to the surface in EITHER
    # direction is what "at least one point is touching" means, and on a tilted foot (toe
    # 2 cm under, heel 5 cm up) the two disagree -- the first says contact is satisfied, the
    # second reports a 2 cm penetration.
    per_node_contact = (strict.to(dt)
                        * (foot_abs - spec.contact_deadband_m).clamp_min(0.0))
    contact_penalty = per_node_contact.sum(dim=2).mean(dim=1)

    # ---- the legacy kernel, kept for comparability only -------------------------------
    kernel = torch.exp(-(foot_clear / spec.hover_sigma_m).pow(2))
    short = (spec.clearance_margin_m - node_sd_min.clamp_max(1e3)).clamp_min(0.0)

    out = {
        # THE two reward terms, both PENALTIES -- wire with NEGATIVE weights.
        "contact_penalty": contact_penalty,
        "penetration_sum_m": acc["sum_m"] / float(f),
        # Diagnostics and alternative reductions.
        "penetration_mean": acc["mean_num"] / acc["mean_den"].clamp_min(1.0),
        "max_penetration_m": max_pen,
        "contact_reward": _masked_mean(kernel, strict, dims=(1, 2)),
        "clearance": short.mean(dim=(1, 2)),
        "support_fraction": contact["support"].to(dt).mean(dim=1),
        "contact_coverage": strict.to(dt).mean(dim=(1, 2)),
        "exempt_fraction": acc["exempt"] / float(f * p),
        "node_contact_penalty": per_node_contact.sum(dim=2),
        "node_penetration_m": node_sum_m,
        "node_sd_min": node_sd_min,
        "node_foot_clearance": foot_clear,
        "node_foot_abs": foot_abs,
        "node_hover": foot_clear.clamp_min(0.0),
        "node_bury": (-foot_clear).clamp_min(0.0),
        "penetration_by_body": by_body / float(f),
    }
    if float(model.areas.sum()) > 0.0:
        out["penetration_volume_m3"] = acc["volume"] / float(f)
        out["penetration_sq"] = acc["sq"] / float(f)
        out["node_penetration_m3"] = node_vol
        out["penetration_volume_m3_served"] = acc["volume_served"] / float(k)

    # ---- the combined total, so every consumer reads one balance -----------------------
    # A PENALTY: higher is worse, and a reward is its negation. Kept as the penalty because
    # that is what the two component terms are, and flipping the sign in one place only (the
    # RewTerm weight, or the viewer's headline) is easier to keep straight than two
    # conventions in the same dict.
    out["weighted_contact"] = spec.contact_weight * out["contact_penalty"]
    out["weighted_penetration"] = spec.penetration_weight * out["penetration_sum_m"]
    out["total_penalty"] = out["weighted_contact"] + out["weighted_penetration"]
    out["node_weighted_contact"] = spec.contact_weight * out["node_contact_penalty"]
    out["node_weighted_penetration"] = spec.penetration_weight * out["node_penetration_m"]
    out["node_total_penalty"] = (out["node_weighted_contact"]
                                 + out["node_weighted_penetration"])

    sl = slice(0, k)
    out["penetration_sum_m_served"] = acc["sum_m_served"] / float(k)
    out["contact_penalty_served"] = per_node_contact[:, sl].sum(dim=2).mean(dim=1)
    out["contact_reward_served"] = _masked_mean(kernel[:, sl], strict[:, sl], dims=(1, 2))
    out["support_fraction_served"] = contact["support"][:, sl].to(dt).mean(dim=1)
    out["contact_coverage_served"] = strict[:, sl].to(dt).mean(dim=(1, 2))
    return out
