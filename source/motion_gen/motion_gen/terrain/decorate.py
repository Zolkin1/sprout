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
    """

    seed: int = 0
    arm_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    tread_full_frac: float = 0.30
    min_tread_m: float = 0.80
    clearance_m: float = 0.02


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
