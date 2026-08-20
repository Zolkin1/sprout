# Copyright 2026 Zachary Olkin. All rights reserved.

"""Rewrite a tile manifest's stair widths, without regenerating any motion.

Terrain is not baked into a clip. The library carries a ``terrain_manifest`` attribute -- a
list of per-tile dicts, each an explicit list of overlay boxes -- and the RL importer, both
MuJoCo emitters and every viewer rebuild the world from it at load. A clip's only tie to its
tile is an integer ``terrain_id`` plus a corner translation. So the ground can be changed
while the motion stays exactly as recorded, which is the whole reason this pass exists.

**Today there is no stair-width variation to speak of.** Measured over the shipped library:
treads run 0.801 to 1.156 m, and approach/landing platforms are ALWAYS exactly ``size_y``
(3.045 to 8.354 m) because all three generators force it. A tracker trained on that sees one
tread width and one platform width, and the 16 x 11 height scan spans +/-0.5 m, so an 0.8 m
flight straddles both edges in every frame.

Each transition tile draws ONE of three arms:

* ``wide_stairs`` -- grow the treads, leave the platform at ``size_y``;
* ``narrow_landing`` -- keep the authored tread, narrow the platform to the tile's own foot
  envelope;
* ``matched`` -- narrow the platform AND bring the treads out to meet it, which is the
  literal "a stair that leads to a platform of the same width" case, and at 1.4 to 3.3 m it
  is far more interesting than matching at full width.

**GROW IN PLACE, NEVER RE-CENTRE.** A tile's ``size_y`` is the bounding box of its CLIPS'
lateral travel, not of its flight, so ``size_y / 2`` is the centroid of where the motion
wandered and the flight sits off it -- by a median 0.149 m and up to 1.765 m. "Widen and
re-centre" therefore moves the tread out from under the feet: on the worst tile it strands
most of the stance samples over void. Every edit here is expressed as a target SPAN, and
:func:`_resize_y` asserts the invariant that makes it safe -- a widened surface is a strict
superset of what it replaced, and a narrowed one still contains the whole foot envelope.

Nothing here consults an RNG at load time: the pass resolves every choice offline, seeded per
``terrain_id``, and writes the result into the manifest. That is the same rule
``terrain_variants`` states -- the generator records resolved parameters, the importer and
the viewers rebuild from them and do not share the generator's RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from motion_gen.terrain.footprint import TileFootprint

#: The three ways a transition tile can be rewritten.
ARMS: tuple[str, ...] = ("wide_stairs", "narrow_landing", "matched")

#: Support shapes a box overlay can take, in the order :attr:`DecorateConfig.box_variant_fracs`
#: weights them. Mirrors ``overlay_geom.OVERLAY_VARIANTS``.
#:
#: The shipped library is 100% ``solid`` -- not by design, but because
#: ``mb_plan_library.py`` passes ``variant_fracs=None`` and the machinery has simply never
#: been switched on. Turning it on is safe in BOTH directions and that is worth being precise
#: about: every variant puts its walkable top at exactly ``base_z + size_z``, and the legs sit
#: strictly inside the solid box's own footprint. So no foot that was supported before is
#: unsupported after, and the height scan reads the same top surface. What changes is what the
#: DEPTH CAMERAS see -- a 0.4 m block becomes a slab on legs, with real void behind it.
BOX_VARIANTS: tuple[str, ...] = ("solid", "floating", "floating_center_leg",
                                 "floating_corner_legs")

#: A box is a TREAD if it spans less than this fraction of the tile, else a PLATFORM.
#: Only ever applied to the ORIGINAL manifest -- see :func:`classify_overlays`.
#: The generators' own rule, inverted: they widen anything whose x-extent exceeds 1.5x the
#: minimum to the full ``size_y``, so today the two classes are perfectly separated -- treads
#: at most 1.156 m, platforms exactly ``size_y``. 0.9 sits in a gap 2 m wide.
TREAD_WIDTH_FRAC = 0.9


@dataclass(frozen=True)
class DecorateConfig:
    """Knobs for the stair-width pass.

    Attributes:
        seed: Base seed. The RNG is re-derived per tile from ``seed`` and ``terrain_id`` so a
            tile's result does not depend on how many tiles preceded it.
        arm_weights: Probability of each entry of :data:`ARMS`.
        tread_full_frac: Fraction of ``wide_stairs`` tiles taken all the way to the tile
            width rather than to an intermediate one. Without it, uniform sampling almost
            never produces the full-width case, which is the one that stops the FM
            overfitting to thin flights.
        min_tread_m: A widened tread is never asked to be narrower than this.
        clearance_m: Slack left inside the tile edge when a surface is grown to it.
        box_variant_fracs: How the box tiles' overlays are split across
            :data:`BOX_VARIANTS`. Must sum to 1.
        box_widen_frac: Fraction of BOX tiles whose overlays are widened at all. The rest
            keep their authored width, so the narrow-target case does not disappear.
        box_full_frac: Of those widened, the fraction taken all the way to the tile.
        wall_frac: Probability a given tile SIDE gets a wall. Two sides per tile, so the
            expected count is twice this.
        wall_height_range: Wall height (m).
        wall_thickness_range: Wall thickness (m).
        wall_window_frac: Probability a wall carries a window hole.
        wall_clearance_m: Gap left between a wall's inner face and the foot envelope.
    """

    seed: int = 0
    arm_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    tread_full_frac: float = 0.30
    min_tread_m: float = 0.80
    clearance_m: float = 0.02
    box_variant_fracs: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20)
    box_widen_frac: float = 0.55
    box_full_frac: float = 0.30
    wall_frac: float = 0.45
    wall_height_range: tuple[float, float] = (1.4, 3.0)
    wall_thickness_range: tuple[float, float] = (0.08, 0.20)
    wall_window_frac: float = 0.45
    wall_clearance_m: float = 0.15


@dataclass
class TileReport:
    """What the pass did to one tile, for the summary and for inspection.

    Attributes:
        terrain_id: The tile.
        arm: Which arm it drew.
        tread_before: Tread widths before (m).
        tread_after: Tread widths after (m).
        platform_before: Platform widths before (m).
        platform_after: Platform widths after (m).
        roles: Role of each overlay, in manifest order, resolved BEFORE the edit. Anything
            validating the result must use these rather than re-deriving from the widths,
            which the edit has just changed.
    """

    terrain_id: int
    arm: str
    roles: list[str] = field(default_factory=list)
    tread_before: list[float] = field(default_factory=list)
    tread_after: list[float] = field(default_factory=list)
    platform_before: list[float] = field(default_factory=list)
    platform_after: list[float] = field(default_factory=list)


def has_width_variation(manifest: list[dict], kind: str = "transition",
                        tol: float = 1e-3) -> bool:
    """Whether a manifest's platforms vary in width at all.

    Straight out of generation they do not: ``mb_terrain_clips`` and ``mb_assemble_library``
    both stretch every wide stair platform to exactly ``size_y``, as a safety net against a
    drifting segment stepping off a landing. That is correct there and it is also why a
    regenerated library silently returns to a uniform world unless this pass is re-run.

    So the question "has this library been decorated?" has a structural answer as well as an
    attribute one, and this is the structural half -- it still works on a manifest that has
    been copied out of its file, and it cannot be fooled by an attribute somebody set by hand.

    Args:
        manifest: The tile list.
        kind: Tile ``type`` to inspect.
        tol: Two widths within this are the same width (m).

    Returns:
        ``True`` if any platform on any tile of that type is not its tile's full width.
    """
    for tile in manifest:
        if tile.get("type") != kind:
            continue
        size_y = float(tile["size_y"])
        for overlay, role in zip(tile["overlays"], classify_overlays(tile)):
            if role != "platform":
                continue
            if abs(float(overlay["size_xyz"][1]) - size_y) > tol:
                return True
    return False


def tile_rng(seed: int, terrain_id: int) -> np.random.Generator:
    """A generator whose stream depends only on the tile, not on iteration order.

    Args:
        seed: Base seed.
        terrain_id: The tile.

    Returns:
        A seeded generator.
    """
    return np.random.default_rng((int(seed) << 20) ^ (int(terrain_id) + 1))


def classify(overlay: dict, size_y: float, frac: float = TREAD_WIDTH_FRAC) -> str:
    """``"tread"`` or ``"platform"``, by lateral extent relative to the tile."""
    return "tread" if float(overlay["size_xyz"][1]) < frac * float(size_y) else "platform"


def classify_overlays(tile: dict, frac: float = TREAD_WIDTH_FRAC) -> list[str]:
    """Role of every overlay, resolved ONCE from the tile as it stands.

    The classification is by width, and this pass CHANGES widths -- so it is only meaningful
    before any edit. Afterwards a widened tread spans nearly the tile and reads as a
    platform, while a narrowed platform reads as a tread. Re-deriving roles from a decorated
    manifest therefore inspects exactly the wrong overlays, which is how the first version of
    the foot-support check came to pass without examining a single widened tread.

    Args:
        tile: A manifest entry.
        frac: Tread/platform threshold, as a fraction of ``size_y``.

    Returns:
        ``"tread"`` or ``"platform"`` per overlay, in manifest order.
    """
    size_y = float(tile["size_y"])
    return [classify(o, size_y, frac) for o in tile["overlays"]]


def span(overlay: dict) -> tuple[float, float]:
    """The overlay's ``(y_lo, y_hi)`` in the tile frame."""
    half = float(overlay["size_xyz"][1]) / 2.0
    centre = float(overlay["pos_xy"][1])
    return centre - half, centre + half


def _resize_y(overlay: dict, lo: float, hi: float, *, must_contain: tuple[float, float],
              widening: bool) -> None:
    """Move an overlay's lateral span to ``[lo, hi]``, in place, with the safety check.

    Args:
        overlay: The overlay dict; ``pos_xy`` and ``size_xyz`` are rewritten.
        lo: New lower edge (m).
        hi: New upper edge (m).
        must_contain: A span the result has to cover.
        widening: ``True`` if the result must also cover the ORIGINAL span. False for a
            deliberate narrowing, where shrinking past the original is the point.

    Raises:
        ValueError: If the result would not contain what it must, or is not positive.
    """
    if hi - lo <= 0.0:
        raise ValueError(f"degenerate span [{lo}, {hi}] for overlay at {overlay['pos_xy']}")
    old_lo, old_hi = span(overlay)
    need_lo, need_hi = must_contain
    if lo > need_lo + 1e-9 or hi < need_hi - 1e-9:
        raise ValueError(
            f"[{lo:.4f}, {hi:.4f}] does not cover the required [{need_lo:.4f}, "
            f"{need_hi:.4f}] -- the surface would not be under the feet"
        )
    if widening and (lo > old_lo + 1e-9 or hi < old_hi - 1e-9):
        raise ValueError(
            f"widening [{old_lo:.4f}, {old_hi:.4f}] to [{lo:.4f}, {hi:.4f}] is not a "
            "superset; growing must never move a surface out from under the feet"
        )
    overlay["size_xyz"] = [float(overlay["size_xyz"][0]), float(hi - lo),
                           float(overlay["size_xyz"][2])]
    overlay["pos_xy"] = [float(overlay["pos_xy"][0]), float((lo + hi) / 2.0)]


def _wall_boxes(lo_y: float, hi_y: float, x0: float, x1: float, height: float,
                base_z: float, window: tuple[float, float, float, float] | None) -> list[dict]:
    """One wall as axis-aligned boxes, with an optional rectangular window cut out.

    MuJoCo and IsaacLab both build boxes and neither does CSG, so a hole is expressed as the
    up-to-four boxes AROUND it rather than as a subtraction. Degenerate pieces are dropped
    instead of emitted: ``build_overlay`` correctly raises on a zero extent, and a window
    flush with an edge legitimately produces one.

    Args:
        lo_y: Wall's near y face (m).
        hi_y: Wall's far y face (m).
        x0: Downrange start (m).
        x1: Downrange end (m).
        height: Wall height above ``base_z`` (m).
        base_z: Bottom of the wall (m).
        window: ``(wx0, wx1, wz0, wz1)`` hole in tile-x and height, or ``None``.

    Returns:
        Overlay dicts, all marked ``decor``.
    """
    def box(bx0: float, bx1: float, bz0: float, bz1: float) -> dict | None:
        if bx1 - bx0 < 1e-3 or bz1 - bz0 < 1e-3:
            return None
        return {
            "kind": "box", "decor": True, "walkable_top": False, "yaw": 0.0,
            "variant": "solid",
            "size_xyz": [float(bx1 - bx0), float(hi_y - lo_y), float(bz1 - bz0)],
            "pos_xy": [float((bx0 + bx1) / 2.0), float((lo_y + hi_y) / 2.0)],
            "base_z": float(bz0),
        }

    top = base_z + height
    if window is None:
        return [b for b in (box(x0, x1, base_z, top),) if b]
    wx0, wx1, wz0, wz1 = window
    wx0, wx1 = max(x0, wx0), min(x1, wx1)
    wz0, wz1 = max(base_z, wz0), min(top, wz1)
    pieces = [
        box(x0, wx0, base_z, top),       # before the hole, full height
        box(wx1, x1, base_z, top),       # after it
        box(wx0, wx1, base_z, wz0),      # under it
        box(wx0, wx1, wz1, top),         # over it
    ]
    return [b for b in pieces if b]


def add_walls(tile: dict, footprint: TileFootprint, cfg: DecorateConfig) -> int:
    """Put up to two VISUAL-ONLY walls along a tile's long edges. Returns how many boxes.

    Why walls at all: at the far camera's 15 degree pitch, 26.9% of its pixels return the
    range sentinel on flat ground -- a quarter of the frame spent on nothing. Vertical
    structure is what fills it, and a stairwell or corridor is what a real stair usually sits
    in, so it is the honest thing to put there.

    Why VISUAL-ONLY: the tracker follows a reference, it does not plan around obstacles. A
    collidable wall the reference's arms sweep through would block a wrist, fire
    ``ee_pose_error`` at 0.3 m and end the episode for a reason the policy cannot fix --
    with ``undesired_contacts`` at -0.1 against ``clf_reward`` at 10 far too weak to teach
    avoidance. It would also be useless to the height scan, which clips at +/-1 m and reads
    a 1.5 m and a 5 m wall as the same number.

    Placement is OUTSIDE the foot envelope by :attr:`DecorateConfig.wall_clearance_m` and
    inside the tile's own footprint. Both matter: outside the envelope so a wall never
    visually swallows the motion, and inside the tile because IsaacLab's tile floor spans the
    CLASS footprint while MuJoCo's spans the tile's own extents -- geometry beyond the tile
    would sit over ground in one sim and over void in the other.

    Args:
        tile: A manifest entry; walls are appended to its overlays.
        footprint: The tile's foot envelope.
        cfg: Knobs.

    Returns:
        Number of overlay boxes added.
    """
    # Its own stream, like the box variants: a tile's walls must not be correlated with
    # which width arm it drew.
    rng = tile_rng(cfg.seed ^ 0x2A11, int(tile["terrain_id"]))
    size_x, size_y = float(tile["size_x"]), float(tile["size_y"])
    added = 0
    for side in (0, 1):
        if rng.random() >= cfg.wall_frac:
            continue
        thickness = float(rng.uniform(*cfg.wall_thickness_range))
        height = float(rng.uniform(*cfg.wall_height_range))
        if side == 0:
            hi_y = footprint.y_min_sized - cfg.wall_clearance_m
            lo_y = hi_y - thickness
            if lo_y < 0.0:
                continue                      # no clear band on this side
        else:
            lo_y = footprint.y_max_sized + cfg.wall_clearance_m
            hi_y = lo_y + thickness
            if hi_y > size_y:
                continue
        x0 = float(rng.uniform(0.0, 0.25 * size_x))
        x1 = float(rng.uniform(0.75 * size_x, size_x))
        window = None
        if rng.random() < cfg.wall_window_frac and (x1 - x0) > 2.0:
            w = float(rng.uniform(0.8, min(2.5, (x1 - x0) * 0.4)))
            cx = float(rng.uniform(x0 + w, x1 - w))
            z0 = float(rng.uniform(0.6, max(0.7, height - 0.9)))
            window = (cx - w / 2.0, cx + w / 2.0, z0, min(z0 + 1.0, height - 0.05))
        boxes = _wall_boxes(lo_y, hi_y, x0, x1, height, 0.0, window)
        tile["overlays"].extend(boxes)
        added += len(boxes)
    return added


def decorate_box_tile(tile: dict, cfg: DecorateConfig,
                     footprint: TileFootprint | None = None) -> str:
    """Give one box tile's overlays a support variant and, sometimes, a wider top.

    Two independent edits.

    **Variant** touches no geometry at all: every variant puts its walkable top at exactly
    ``base_z + size_z`` and keeps its legs inside the solid box's own footprint, so the
    surface the feet meet is identical and only what is underneath changes.

    **Width** is the same grow-in-place rule the stairs get, and it exists because the boxes
    are all NARROW. Measured over the shipped library: 0.264 to 1.197 m, a median of 23% of
    their tile and a maximum of 37%, with not one spanning its tile. So the robot only ever
    jumps onto a narrow target -- there is no wide platform to land on anywhere in training.
    Widening only ever ADDS surface, so a recorded landing cannot be invalidated; ``_resize_y``
    asserts that rather than assuming it.

    Args:
        tile: A manifest entry of type ``box``; its overlays are edited.
        cfg: Knobs.
        footprint: The tile's foot envelope. Required to widen; without it only the variant
            is assigned, because a widened box must still be a superset of where the feet go.

    Returns:
        A summary for the run log.

    Raises:
        ValueError: If the variant weights do not sum to 1.
    """
    if abs(sum(cfg.box_variant_fracs) - 1.0) > 1e-9:
        raise ValueError(
            f"box_variant_fracs must sum to 1, got {sum(cfg.box_variant_fracs)}"
        )
    rng = tile_rng(cfg.seed ^ 0x5EED, int(tile["terrain_id"]))
    chosen = []
    for overlay in tile["overlays"]:
        variant = str(rng.choice(BOX_VARIANTS, p=list(cfg.box_variant_fracs)))
        overlay["variant"] = variant
        chosen.append(variant)

    widened = "asis"
    if footprint is not None and rng.random() < cfg.box_widen_frac:
        size_y = float(tile["size_y"])
        full_lo, full_hi = cfg.clearance_m, size_y - cfg.clearance_m
        env_lo = max(full_lo, footprint.y_min_sized)
        env_hi = min(full_hi, footprint.y_max_sized)
        for overlay in tile["overlays"]:
            old_lo, old_hi = span(overlay)
            need_lo, need_hi = min(old_lo, env_lo), max(old_hi, env_hi)
            if rng.random() < cfg.box_full_frac:
                target_lo, target_hi = full_lo, full_hi
            else:
                t = float(rng.random())
                target_lo = need_lo + (full_lo - need_lo) * t
                target_hi = need_hi + (full_hi - need_hi) * t
            _resize_y(overlay, min(target_lo, old_lo), max(target_hi, old_hi),
                      must_contain=(need_lo, need_hi), widening=True)
        widened = "widened"
    return f"{','.join(chosen)}|{widened}"


def decorate_transition_tile(tile: dict, footprint: TileFootprint,
                             cfg: DecorateConfig) -> TileReport:
    """Rewrite one transition tile's stair widths in place. Returns what changed.

    Args:
        tile: A manifest entry; its overlays are edited.
        footprint: The tile's measured foot envelope.
        cfg: Knobs.

    Returns:
        The per-tile report.

    Raises:
        ValueError: If an edit would leave a surface not covering the feet.
    """
    rng = tile_rng(cfg.seed, int(tile["terrain_id"]))
    size_y = float(tile["size_y"])
    arm = str(rng.choice(ARMS, p=list(cfg.arm_weights)))
    report = TileReport(terrain_id=int(tile["terrain_id"]), arm=arm)

    # The envelope, clipped into the tile. A tile whose motion runs past its own edge cannot
    # be narrowed to fit it, and the clip is what makes that a no-op rather than an error.
    env_lo = max(0.0 + cfg.clearance_m, footprint.y_min_sized)
    env_hi = min(size_y - cfg.clearance_m, footprint.y_max_sized)
    if env_hi - env_lo <= 0.0:
        raise ValueError(
            f"terrain {tile['terrain_id']}: the foot envelope "
            f"[{footprint.y_min_sized:.3f}, {footprint.y_max_sized:.3f}] does not intersect "
            f"the tile [0, {size_y:.3f}]. The manifest and the clips disagree about this tile."
        )

    roles = classify_overlays(tile)
    report.roles = roles
    treads = [o for o, r in zip(tile["overlays"], roles) if r == "tread"]
    platforms = [o for o, r in zip(tile["overlays"], roles) if r == "platform"]
    report.tread_before = [float(o["size_xyz"][1]) for o in treads]
    report.platform_before = [float(o["size_xyz"][1]) for o in platforms]

    if arm in ("narrow_landing", "matched"):
        for overlay in platforms:
            old_lo, old_hi = span(overlay)
            # Narrow INTO the existing platform: never grow it, and never past the envelope.
            _resize_y(overlay, max(old_lo, env_lo), min(old_hi, env_hi),
                      must_contain=(env_lo, env_hi), widening=False)

    if arm in ("wide_stairs", "matched"):
        # One target span for every tread on the tile, so a flight stays a flight.
        union_lo = min(span(o)[0] for o in treads) if treads else env_lo
        union_hi = max(span(o)[1] for o in treads) if treads else env_hi
        need_lo, need_hi = min(union_lo, env_lo), max(union_hi, env_hi)
        if arm == "matched":
            target_lo, target_hi = need_lo, need_hi
        else:
            full_lo, full_hi = 0.0 + cfg.clearance_m, size_y - cfg.clearance_m
            if rng.random() < cfg.tread_full_frac:
                target_lo, target_hi = full_lo, full_hi
            else:
                # Somewhere between "just covers what it must" and the whole tile.
                t = float(rng.random())
                target_lo = need_lo + (full_lo - need_lo) * t
                target_hi = need_hi + (full_hi - need_hi) * t
            if target_hi - target_lo < cfg.min_tread_m:
                mid = (target_lo + target_hi) / 2.0
                target_lo, target_hi = mid - cfg.min_tread_m / 2, mid + cfg.min_tread_m / 2
        for overlay in treads:
            old_lo, old_hi = span(overlay)
            _resize_y(overlay, min(target_lo, old_lo), max(target_hi, old_hi),
                      must_contain=(min(need_lo, old_lo), max(need_hi, old_hi)),
                      widening=True)

    report.tread_after = [float(o["size_xyz"][1]) for o in treads]
    report.platform_after = [float(o["size_xyz"][1]) for o in platforms]
    return report
