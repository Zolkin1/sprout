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
        min_tread_m: The narrow end of the stair-width band (m). Every flight is at least
            this wide.
        max_tread_m: The wide end of the band (m). A flight is only ever taken past it to
            keep a foot on the surface, which the traversal envelope never requires.
        clearance_m: Slack left inside the tile edge when a surface is grown to it.
        box_variant_fracs: How the box tiles' overlays are split across
            :data:`BOX_VARIANTS`. Must sum to 1.
        box_widen_frac: Fraction of BOX tiles whose overlays are widened at all. The rest
            keep their authored width, so the narrow-target case does not disappear.
        box_full_frac: Of those widened, the fraction taken all the way to the tile.
        wall_frac: Probability a given tile SIDE gets a wall. FOUR sides per tile, so the
            expected count is four times this and a full room is ``wall_frac ** 4``.
        wall_height_range: Wall height (m).
        wall_thickness_range: Wall thickness (m).
        wall_window_frac: Probability a wall carries a window hole.
        wall_clearance_m: Gap the motion must have from a tile edge before that edge can
            take a wall.
        flat_room_frac: Fraction of flat-tile variants that get a room at all; the rest stay
            open ground, so the FM still sees unbounded flat.
        wall_collidable: Emit the walls as ordinary collidable terrain instead of the
            visual-only channel. OFF by design: the tracker follows a reference and does not
            plan around obstacles, so a wall the reference's arm sweeps through blocks a
            wrist, fires ``ee_pose_error`` at 0.3 m and ends the episode for a reason the
            policy cannot fix -- ``undesired_contacts`` at -0.1 against ``clf_reward`` at 10
            is far too weak to teach avoidance. The walls exist to fill the far camera's
            frame, and they do that either way.
    """

    seed: int = 0
    arm_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    # The band of stair widths to train on: 4 ft to 8 ft, which is what real stairwells are
    # (the ones measured for this robot are 4, 5 and 7 ft). Every flight lands inside it --
    # `tread_full_frac`, which used to send 30% of flights all the way to the tile edge, is
    # gone with the whole-tile arm it served.
    min_tread_m: float = 1.219   # 4 ft
    max_tread_m: float = 2.438   # 8 ft
    clearance_m: float = 0.02
    box_variant_fracs: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20)
    box_widen_frac: float = 0.55
    box_full_frac: float = 0.30
    wall_frac: float = 0.55
    wall_height_range: tuple[float, float] = (1.4, 3.0)
    wall_thickness_range: tuple[float, float] = (0.08, 0.20)
    wall_window_frac: float = 0.45
    wall_clearance_m: float = 0.15
    flat_room_frac: float = 0.75
    wall_collidable: bool = False


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

    The test is ROLE-FREE, and has to be. Roles are classified by width and this pass changes
    widths, so re-deriving them from a decorated manifest inspects the wrong overlays --
    exactly what :func:`classify_overlays`' docstring warns about. The earlier version did that
    and only worked by accident, because widened treads happened to read back as
    not-quite-full-width platforms; once the treads were capped at the stair-width band, every
    overlay on a narrowed tile read as a tread and the whole manifest reported no variation.

    What is true regardless of roles: straight out of generation EVERY transition tile carries
    at least one overlay exactly ``size_y`` wide, because that is what the generators force.
    A tile with none has been narrowed.

    Args:
        manifest: The tile list.
        kind: Tile ``type`` to inspect.
        tol: Two widths within this are the same width (m).

    Returns:
        ``True`` if any tile of that type has no full-width overlay left.
    """
    for tile in manifest:
        if tile.get("type") != kind:
            continue
        size_y = float(tile["size_y"])
        widths = [float(o["size_xyz"][1]) for o in tile["overlays"]]
        if widths and not any(abs(w - size_y) <= tol for w in widths):
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


#: Loader defaults the class footprint is built from. Mirrored here, and CHECKED against
#: the real loader by ``source/robot_rl/tests/test_wall_footprint.py`` -- see
#: :func:`class_footprints` for why this duplication exists and what makes it safe.
DEFAULT_SIZE_MARGIN = 1.0
DEFAULT_TRANSITION_BACK_BUFFER = 1.0

#: Flat-tile margin, per side, beyond the flat clips' own travel.
#:
#: THE one definition. It used to be two: ``assemble_clip_terrain`` defaulted to 2.0 while
#: ``ClipTerrainImporterCfg`` said 20.0, so the viewers -- which call the function directly --
#: drew flat tiles 36 m smaller than the ones training built. Nothing noticed until the flat
#: rooms were sized against the real footprint and promptly hung off the edge of the viewer's
#: smaller one, which reads as walls overlapping the neighbouring tiles.
#:
#: **Known consequence, measured and accepted (2026-08-19).** A flat tile is sized by the
#: LONGEST flat clip (23.03 x 20.65 m of travel) but every clip is re-centred on it by its own
#: bbox, and the median clip travels 5.06 m -- so the median clip sits 27.9 m from the nearest
#: tile-edge wall (p1 23.2, min 23.2). The far camera reaches 11.13 m downrange, so at this
#: buffer a flat room is NEVER in frame: 0.0% of the 3000 flat clips have a wall within reach
#: (2.0 m would give 88.9%, 4.0 m gives 30.7%). The rooms are in the manifest and render in
#: every viewer; they are simply out of sensor range. Zach chose to keep 20.0 rather than
#: shrink the flat tiles. Walls on the stair and box tiles are unaffected -- those tiles are
#: metres across, not tens of metres, so their edge walls are in frame throughout.
DEFAULT_FLAT_BUFFER = 20.0


def flat_tile_footprint(travel_xy: tuple[float, float],
                        flat_buffer: float = DEFAULT_FLAT_BUFFER) -> tuple[float, float]:
    """The floor a flat tile is rendered at, from the flat clips' travel.

    Flat tiles carry no manifest entry, so unlike every other class their size is not written
    down anywhere -- it is computed at load from the motion. This mirrors that computation so
    the decoration pass can place walls on the tile's actual edge.

    Args:
        travel_xy: Max ``(x, y)`` span of any flat clip's pelvis (m).
        flat_buffer: Margin per side (m).

    Returns:
        ``(footprint_x, footprint_y)``.
    """
    return (float(travel_xy[0]) + 2.0 * flat_buffer,
            float(travel_xy[1]) + 2.0 * flat_buffer)


def class_footprints(manifest: list[dict], size_margin: float = DEFAULT_SIZE_MARGIN,
                     transition_back_buffer: float = DEFAULT_TRANSITION_BACK_BUFFER,
                     ) -> dict[str, tuple[float, float]]:
    """The floor each tile CLASS is actually rendered at, by type.

    A tile's ``size_x`` / ``size_y`` is not the ground the robot sees. The loader pads every
    tile of a class to one common footprint -- ``max(size_x) + margin`` over the whole class
    -- so IsaacLab can lay them out on a uniform grid. Measured on the shipped library, that
    is about 1 m of slack for terrain tiles but 4.9 to 7.1 m in y for transition and box
    tiles: a 3.6 m-wide box tile is rendered on a 10.7 m floor.

    So anything meant to sit at the EDGE of what you can see has to use this, not the
    manifest size. Walls placed at ``size_y`` end up metres inside the floor, which is
    exactly how they first shipped.

    The geometry stays corner-anchored at ``(0, 0)`` and the slack is all on the high side,
    so the low edges coincide and only the high ones move.

    This duplicates ``clip_terrain_loader._class_fp`` because the decoration pass resolves
    positions offline and IsaacLab is not importable from ``motion_gen``. What makes that
    safe is not care -- it is ``test_wall_footprint.py``, which builds the real layout and
    fails if the two ever disagree.

    Args:
        manifest: The tile list.
        size_margin: Extra flat added to every class footprint (m).
        transition_back_buffer: Extra x on transition tiles, behind the spawn (m).

    Returns:
        Tile type -> ``(footprint_x, footprint_y)``. Types with no tiles are omitted.
    """
    out: dict[str, tuple[float, float]] = {}
    for kind in {str(m.get("type")) for m in manifest}:
        tiles = [m for m in manifest if m.get("type") == kind]
        if not tiles:
            continue
        fx = max(float(m["size_x"]) for m in tiles) + size_margin
        fy = max(float(m["size_y"]) for m in tiles) + size_margin
        if kind == "transition":
            fx += transition_back_buffer
        out[kind] = (fx, fy)
    return out


def _wall_segments(axis: str, lo_y: float, hi_y: float, x0: float, x1: float,
                   height: float, base_z: float,
                   window: tuple[float, float, float, float] | None) -> list[dict]:
    """One wall as axis-aligned boxes, with an optional rectangular window cut out.

    MuJoCo and IsaacLab both build boxes and neither does CSG, so a hole is expressed as the
    up-to-four boxes AROUND it rather than as a subtraction. Degenerate pieces are dropped
    instead of emitted: ``build_overlay`` correctly raises on a zero extent, and a window
    flush with an edge legitimately produces one.

    Args:
        axis: ``"y"`` for a wall running along x (on a y edge), ``"x"`` for one running along
            y. The window is cut along the wall's LENGTH, so which axis that is matters.
        lo_y: Wall's low y face (m).
        hi_y: Wall's high y face (m).
        x0: Wall's low x face (m).
        x1: Wall's high x face (m).
        height: Wall height above ``base_z`` (m).
        base_z: Bottom of the wall (m).
        window: ``(along0, along1, z0, z1)`` hole, measured along the wall's length from its
            low end, or ``None``.

    Returns:
        Overlay dicts. The caller marks them ``decor``.
    """
    def box(ax0: float, ax1: float, bz0: float, bz1: float) -> dict | None:
        """One piece, spanning ``[ax0, ax1]`` along the wall and ``[bz0, bz1]`` in z."""
        if ax1 - ax0 < 1e-3 or bz1 - bz0 < 1e-3:
            return None
        if axis == "y":
            sx, sy = ax1 - ax0, hi_y - lo_y
            px, py = (ax0 + ax1) / 2.0, (lo_y + hi_y) / 2.0
        else:
            sx, sy = x1 - x0, ax1 - ax0
            px, py = (x0 + x1) / 2.0, (ax0 + ax1) / 2.0
        return {
            "kind": "box", "walkable_top": False, "yaw": 0.0, "variant": "solid",
            "size_xyz": [float(sx), float(sy), float(bz1 - bz0)],
            "pos_xy": [float(px), float(py)],
            "base_z": float(bz0),
        }

    a0, a1 = (x0, x1) if axis == "y" else (lo_y, hi_y)
    top = base_z + height
    if window is None:
        return [b for b in (box(a0, a1, base_z, top),) if b]
    # The window is given along the wall from its low end.
    w0, w1 = a0 + window[0], a0 + window[1]
    w0, w1 = max(a0, w0), min(a1, w1)
    wz0, wz1 = max(base_z, window[2]), min(top, window[3])
    pieces = [
        box(a0, w0, base_z, top),        # before the hole, full height
        box(w1, a1, base_z, top),        # after it
        box(w0, w1, base_z, wz0),        # under it
        box(w0, w1, wz1, top),           # over it
    ]
    return [b for b in pieces if b]


def add_walls(tile: dict, footprint: TileFootprint, cfg: DecorateConfig,
              tile_fp: tuple[float, float] | None = None) -> tuple[int, int]:
    """Wall a tile's EDGES, up to all four, so it reads as a room.

    Returns ``(sides, boxes)``. The two differ: a wall with a window is up to FOUR boxes, so
    counting boxes to answer "how many walls does this tile have" overstates it by up to 4x.

    Why walls: at the far camera's 15 degree pitch, 26.9% of its pixels return the range
    sentinel on flat ground -- a quarter of the frame spent on nothing. Vertical structure is
    what fills it, and a room is what a real stair or step usually sits in.

    **On the EDGE, always.** An earlier version placed each wall a fixed clearance outside
    the FOOT ENVELOPE, which is only the tile edge when the motion happens to fill the tile.
    On a pure-stair tile with 3.8 m of unused width it put the wall through the middle of the
    tile instead, which reads as a random slab rather than a room. The edge is the tile's
    own ``[0, size_x] x [0, size_y]``, and the envelope is now only used to decide WHETHER a
    side can take a wall -- never where it goes.

    **The edge is the FLOOR's, not the manifest tile's.** Those differ by 4.9 to 7.1 m in y
    on transition and box tiles, because the loader pads every tile of a class to one common
    footprint (:func:`class_footprints`). Using ``size_y`` puts the wall metres inside the
    visible ground -- which is how this first shipped, and reads as a slab standing in a
    field rather than a room.

    A side is skipped when the motion reaches it. That is what stops a wall appearing across
    the mouth of a corridor: a transition tile is traversed along +x, so its x-extremes are
    where the robot enters and leaves, and those two sides almost never qualify.

    Args:
        tile: A manifest entry; walls are appended to its overlays.
        footprint: The tile's foot envelope, used only as a veto.
        cfg: Knobs.
        tile_fp: The floor this tile is rendered at, from :func:`class_footprints`. Falls
            back to the manifest size, which is only correct when the two coincide.

    Returns:
        ``(number of walled sides, number of overlay boxes)``.
    """
    # Its own stream, like the box variants: a tile's walls must not be correlated with
    # which width arm it drew.
    rng = tile_rng(cfg.seed ^ 0x2A11, int(tile["terrain_id"]))
    if tile_fp is None:
        size_x, size_y = float(tile["size_x"]), float(tile["size_y"])
    else:
        size_x, size_y = float(tile_fp[0]), float(tile_fp[1])
    clear = cfg.wall_clearance_m

    # (axis, at the low edge?) -- y walls run along x, x walls run along y.
    sides = (("y", True), ("y", False), ("x", True), ("x", False))
    added, n_sides = 0, 0
    thickness = float(rng.uniform(*cfg.wall_thickness_range))
    height = float(rng.uniform(*cfg.wall_height_range))
    for axis, low in sides:
        if rng.random() >= cfg.wall_frac:
            continue
        if axis == "y":
            # Room for a wall only if the motion keeps clear of this edge.
            if low and footprint.y_min_sized < thickness + clear:
                continue
            if not low and footprint.y_max_sized > size_y - thickness - clear:
                continue
            lo_y = 0.0 if low else size_y - thickness
            hi_y = thickness if low else size_y
            x0, x1 = 0.0, size_x
        else:
            if low and footprint.x_min < thickness + clear:
                continue
            if not low and footprint.x_max > size_x - thickness - clear:
                continue
            lo_y, hi_y = 0.0, size_y
            x0 = 0.0 if low else size_x - thickness
            x1 = thickness if low else size_x

        window = None
        length = (x1 - x0) if axis == "y" else (hi_y - lo_y)
        if rng.random() < cfg.wall_window_frac and length > 2.5:
            w = float(rng.uniform(0.8, min(2.5, length * 0.35)))
            centre = float(rng.uniform(w, length - w))
            z0 = float(rng.uniform(0.6, max(0.7, height - 0.9)))
            window = (centre - w / 2.0, centre + w / 2.0, z0,
                      min(z0 + 1.0, height - 0.05))

        boxes = _wall_segments(axis, lo_y, hi_y, x0, x1, height, 0.0, window)
        for b in boxes:
            b["decor"] = not cfg.wall_collidable
        tile["overlays"].extend(boxes)
        added += len(boxes)
        n_sides += 1
    return n_sides, added


def make_flat_rooms(n_variants: int, travel_xy: tuple[float, float],
                    cfg: DecorateConfig,
                    flat_buffer: float = DEFAULT_FLAT_BUFFER) -> list[list[dict]]:
    """Build room variants for the FLAT tiles. Returns one overlay list per variant.

    Flat tiles are 30% of training and the only place the robot walks on open ground, so they
    are where "deployed in a room" is actually learned -- and undecorated they are a bare
    plane out to a 20 m buffer, which gives the far camera nothing at all to look at.

    **Walls go on the tile EDGE**, the same rule every other tile type follows. An earlier
    version stood them off the MOTION instead, at 3-8 m, which put a 29 x 31 m room inside a
    63 x 61 m tile -- 16 m of bare floor outside the walls on every side, so the room read as
    a fence in a field with the neighbouring tile visible past it.

    At the shipped :data:`DEFAULT_FLAT_BUFFER` those walls are beyond the far camera's reach;
    see that constant for the measurement and why it stands.

    **Positions are relative to the tile CENTRE.** Flat tiles have no manifest entry: the
    loader synthesises them and sizes them from the motion, so there is no authored corner to
    anchor to. The loader adds the centre back.

    Args:
        n_variants: How many distinct rooms to author. The loader hands them to flat tiles
            round-robin, which keeps the pooling property -- any flat clip may land on any
            flat tile -- while still varying what the cameras see.
        travel_xy: The flat clips' own ``(x, y)`` travel span (m). Must be the max over ALL
            flat clips: it is a maximum, so a sample under-reports it (200 of 3000 clips
            missed 4 m of it) and the room would then be built inside the floor again.
        cfg: Knobs.
        flat_buffer: Margin per side; see :data:`DEFAULT_FLAT_BUFFER`.

    Returns:
        One list of overlay dicts per variant, each already marked as decoration. An empty
        list is a legitimate variant: open ground.
    """
    fx, fy = flat_tile_footprint(travel_xy, flat_buffer)
    half_x, half_y = fx / 2.0, fy / 2.0
    out: list[list[dict]] = []
    for i in range(n_variants):
        rng = tile_rng(cfg.seed ^ 0xF1A7, i)
        if rng.random() >= cfg.flat_room_frac:
            out.append([])
            continue
        thickness = float(rng.uniform(*cfg.wall_thickness_range))
        height = float(rng.uniform(*cfg.wall_height_range))
        overlays: list[dict] = []
        for axis, sign in (("y", -1.0), ("y", 1.0), ("x", -1.0), ("x", 1.0)):
            if rng.random() >= cfg.wall_frac:
                continue
            if axis == "y":
                lo_y = -half_y if sign < 0 else half_y - thickness
                hi_y = lo_y + thickness
                x0, x1 = -half_x, half_x
            else:
                lo_y, hi_y = -half_y, half_y
                x0 = -half_x if sign < 0 else half_x - thickness
                x1 = x0 + thickness
            length = (x1 - x0) if axis == "y" else (hi_y - lo_y)
            window = None
            if rng.random() < cfg.wall_window_frac and length > 3.0:
                w = float(rng.uniform(1.0, min(3.0, length * 0.3)))
                centre = float(rng.uniform(w, length - w))
                z0 = float(rng.uniform(0.6, max(0.7, height - 1.2)))
                window = (centre - w / 2.0, centre + w / 2.0, z0,
                          min(z0 + 1.4, height - 0.05))
            boxes = _wall_segments(axis, lo_y, hi_y, x0, x1, height, 0.0, window)
            for b in boxes:
                b["decor"] = not cfg.wall_collidable
            overlays.extend(boxes)
        out.append(overlays)
    return out


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

    # ONE CENTRE for the flight and its platforms, and the platforms are never narrower than the
    # flight. Before this the two were placed against DIFFERENT envelopes -- treads about their own
    # flight centre, platforms narrowed into the whole-clip envelope -- so they drifted apart:
    # measured on g1_mb_v6c, a median 0.190 m centre offset (max 0.742) with the platform NARROWER
    # than the stair on 3 of 66 tiles. A staircase stepping onto a landing that is offset from it,
    # or narrower than it, is not a thing the robot should be asked to walk.
    #
    # The centre is the TRAVERSAL envelope's, because that is where the feet actually cross the
    # treads. Forcing the platform to share it costs width -- it has to reach twice the larger half
    # of an asymmetric approach envelope -- and that is the right direction to overpay in: a wider
    # landing is always safe, a narrower one drops a foot.
    centre = sum(footprint.tread_envelope()) / 2.0

    if treads:
        # EVERY flight lands inside [min_tread_m, max_tread_m] -- the band of real stair widths
        # the robot is to be trained on. The arm chooses WHERE in the band; it no longer chooses
        # whether the flight is in it at all, and no arm reaches the whole tile any more.
        #
        # The floor a flight must carry is its TRAVERSAL envelope, not the whole clip's. Measured
        # on the v6b build: the feet that are actually on the treads span 0.35 m median / 0.56 m
        # max per tile, while the whole-clip union over a tile's 50 clips is 2.51 m. The extra is
        # the approach and exit wandering ~1 m BETWEEN clips on the flat platforms -- which the
        # platforms carry, and which is why sizing a flight from `env_lo/env_hi` made every
        # staircase about four times wider than anything ever walked on it.
        tread_lo, tread_hi = footprint.tread_envelope()
        need_lo = max(0.0 + cfg.clearance_m, min(tread_lo, env_hi))
        need_hi = min(size_y - cfg.clearance_m, max(tread_hi, env_lo))
        if need_hi - need_lo <= 0.0:
            raise ValueError(
                f"terrain {tile['terrain_id']}: the traversal foot envelope "
                f"[{tread_lo:.3f}, {tread_hi:.3f}] does not intersect the tile "
                f"[0, {size_y:.3f}]."
            )

        if arm == "matched":
            width = cfg.min_tread_m          # flight and narrowed landing end up alike
        else:
            width = float(rng.uniform(cfg.min_tread_m, cfg.max_tread_m))
        # The band is a target, never a promise to under-run the feet or over-run the tile.
        width = min(max(width, need_hi - need_lo), size_y - 2.0 * cfg.clearance_m)

        # Centred on the traversal feet, then slid by the least that keeps it inside the tile.
        target_lo = centre - width / 2.0
        target_lo = min(max(target_lo, need_hi - width), need_lo)
        target_lo = min(max(target_lo, cfg.clearance_m), size_y - cfg.clearance_m - width)
        target_hi = target_lo + width

        for overlay in treads:
            # widening=False: a flight may now SHRINK, and the check that keeps it honest is
            # that it still covers the feet that walked on it.
            _resize_y(overlay, target_lo, target_hi,
                      must_contain=(need_lo, need_hi), widening=False)

    # The platforms, AFTER the flight, so its final span is known and can be a floor. They carry
    # the approach and exit wander (the whole-clip envelope), the flight does not.
    if platforms:
        flight_lo, flight_hi = ((min(span(o)[0] for o in treads), max(span(o)[1] for o in treads))
                                if treads else (env_lo, env_hi))
        c = (flight_lo + flight_hi) / 2.0 if treads else centre
        # Half-width that (a) covers the flight, (b) covers the whole-clip feet about the SAME
        # centre. Widening the narrow side rather than re-centring is what keeps them aligned.
        half = max(flight_hi - c, c - flight_lo, env_hi - c, c - env_lo)
        if arm == "wide_stairs":
            # This arm leaves the landings alone; it only has to satisfy the two floors.
            half = max(half, *( [max(span(o)[1] - c, c - span(o)[0]) for o in platforms] or [0.0] ))
        half = min(half, min(c - cfg.clearance_m, size_y - cfg.clearance_m - c))
        for overlay in platforms:
            _resize_y(overlay, c - half, c + half,
                      must_contain=(max(env_lo, c - half), min(env_hi, c + half)), widening=False)

    report.tread_after = [float(o["size_xyz"][1]) for o in treads]
    report.platform_after = [float(o["size_xyz"][1]) for o in platforms]
    return report
