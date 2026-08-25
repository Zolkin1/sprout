# Copyright 2026 Zachary Olkin. All rights reserved.

"""Visual clutter for a tile: geometry the cameras see and nothing else touches.

The generator is trained on a world with almost nothing in it. Every wall in the shipped
library sits at a tile edge the motion never approaches, the flat tiles are bare plane out to
a 20 m buffer, and the one close-range obstacle family is excluded from FM training by
default. So its entire experience of tall geometry inside camera range is *a box it is about
to jump onto* or *a staircase it is about to climb* -- and the probe
(``scripts/rsl_rl/fm_terrain_probe.py``) measures the consequence: a wall it has no reason to
react to displaces its predicted path further than a real staircase does.

This pass adds the missing geometry. Everything it emits is ``decor: True``, so it reaches the
depth cameras and reaches nothing else -- no collider, absent from the height scan, outside
the ground mesh and outside the tile layout. That is what lets it be added to an EXISTING
library and re-rendered against an EXISTING recording: the tiles do not move, so the recorded
world poses stay valid, and only the images change.

**The placement rule is the approach zone, and it is what keeps the lesson honest.**

Clutter divides into two tiers by whether the robot should ever act on it:

* **Non-interactable** -- walls, pillars, overhangs, crates. There is no correct reaction, so
  the only constraint is that the robot never walks into one. These may be CLOSE, which
  matters because close is the regime that is missing.
* **Interactable-looking** -- a decoy box or staircase. Here proximity is not enough. Each
  real terrain feature has an APPROACH ZONE, the distance within which a robot moving toward
  it is expected to latch on and start the interaction; a decoy inside that zone would be a
  window where the robot is committed to terrain and the reference does nothing, which
  teaches exactly the ambiguity this is meant to remove. So a decoy must sit with its whole
  approach zone clear of the path. The robot may walk PAST one -- that is the lesson.

**The veto is computed against the CLOSED-LOOP RECORDING, never against the clips.** This is
the whole correctness argument for the pass. The clip is what the robot was ASKED to do; the
recording is where it actually went, and the two differ -- the tracker has its own error, and
the reference is ego-anchored laterally at runtime and drifts from its reset placement by a
p99 of 0.21 m (:data:`~motion_gen.terrain.footprint.EGO_ANCHOR_ALLOWANCE_M`) on top of that.
Clutter fitted to the clips is clutter the real robot walks through, and since the re-render
replays the RECORDED poses, that is exactly the trajectory the boxes must miss.

It is also an OCCUPANCY GRID rather than an envelope, and that is not a refinement. A tile
hosts many clips and a closed-loop rollout of all of them covers most of its area, so a
bounding box would veto every candidate and the pass would emit nothing. What the grid says
is "the robot was never HERE", which is both true and permissive enough to place against.

Everything is an axis-aligned box. Both loaders refuse a rotated overlay
(``OverlayVolumeCfg`` is axis-aligned and the MuJoCo emitter would rotate it), so staying
axis-aligned is what buys Isaac / MuJoCo / viser parity for free.

numpy only -- no IsaacLab, no torch -- so this loads in the generation envs as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


#: How far ahead of a terrain feature a robot moving toward it is expected to latch on (m).
#:
#: One declared number, not a fitted one, and deliberately generous. It exists so that
#: "is this decoy far enough from the path?" has a single answer that the clip generator can
#: later be made to honour -- the same distance at which a clip should stop tracking the
#: steady-state command and start following the terrain motion. Today the clips have no such
#: contract (the latch is abrupt and its extent varies by an order of magnitude between
#: families), so this is the definition arriving before the thing it will govern.
#:
#: A decoy is vetoed if the path enters its zone AT ALL, so an over-estimate costs placement
#: opportunities while an under-estimate costs the training signal. Prefer the former.
APPROACH_ZONE_M: float = 2.5

#: Half-width of the robot in plan, for the "never walk into it" veto (m).
#:
#: MEASURED, not assumed: forward kinematics of the recorded joint configurations puts the
#: furthest link from the pelvis at p50 0.28, p90 0.35, p99 0.46 and max 0.48 m -- the arms
#: and hands, which swing well outside the torso. An earlier 0.35 was the p90, so it cleared
#: the body on only 9 poses in 10 and left ~2 cm to the robot's skin at the worst of them.
#: Rounded up past the measured max.
#:
#: The occupancy this is applied to holds the PELVIS position, which is why the radius has to
#: cover the whole body rather than the base.
BODY_RADIUS_M: float = 0.50

#: Extra clearance on every veto (m), beyond the body itself.
#:
#: Deliberately modest. The body radius above already covers the robot's own widest reach,
#: so this is margin on top of that, and margin is not free here: clutter exists to be SEEN,
#: and pushing it further out moves it away from the band where the far camera resolves
#: anything. The sample-gap argument that once justified a larger value no longer applies --
#: the occupancy interpolates along each episode rather than binning isolated windows, so
#: there is no hole between consecutive poses for a box to sit in.
CLEARANCE_M: float = 0.15

#: Classes the robot must never act on. Named in the overlay as ``clutter_class`` so the
#: re-render can tell new clutter from the walls the decoration pass already placed.
NON_INTERACTABLE: tuple[str, ...] = ("wall", "pillar", "overhang", "crate",
                                     "step_decoy", "background")

#: Classes that look like terrain and therefore carry an approach zone.
INTERACTABLE: tuple[str, ...] = ("decoy_box", "decoy_stairs")


@dataclass
class ClutterConfig:
    """Knobs for one clutter pass.

    Attributes:
        seed: Base seed; every tile draws from its own stream.
        wall_frac: Probability a tile gets a free-standing wall across its far approach.
        wall_size_range: ``(height_lo, height_hi)`` for walls (m). Height is barely
            observable -- the frame's top edge cuts a nearby wall well below its top -- so
            this varies mostly for the sake of the MuJoCo and viser views.
        pillar_count_range: Pillars per tile, placed beside the corridor.
        overhang_frac: Probability of a soffit spanning the corridor above head height.
        overhang_z_range: Underside height (m). Must clear the tallest recorded body point.
        crate_count_range: Large masses per tile, beside the corridor.
        step_decoy_frac: Probability of a step-height ledge beside the corridor. Kept small:
            the probe measured a lone riser moving the plan by millimetres, so this is cheap
            coverage rather than a lever.
        decoy_frac: Probability of an interactable-looking decoy, subject to the approach-zone
            veto (which will reject most placements on a narrow tile).
        background_count_range: Distant masses, purely to give the far camera content.
        background_range_m: How far out they go (m).
        cylinder_frac_pillar: Fraction of pillars emitted as cylinders. High, because a
            pillar is the one class that is round in the real world as often as not.
        cylinder_frac_other: Fraction of crates and background masses emitted as cylinders.
            Low on purpose -- mostly boxes, a few round things, which is what a room looks
            like and what keeps the depth silhouettes varied without making the set exotic.
        flat_repeats: How many times the per-tile draw is repeated on a FLAT tile. The other
            counts are sized for a terrain tile of a few metres; a flat tile is 63 x 61 m,
            roughly two orders of magnitude more area, and it is the case that most needs
            filling -- so one draw leaves it essentially as bare as before.
    """

    seed: int = 0
    wall_frac: float = 0.55
    wall_size_range: tuple[float, float] = (1.2, 3.0)
    pillar_count_range: tuple[int, int] = (0, 3)
    overhang_frac: float = 0.25
    overhang_z_range: tuple[float, float] = (2.1, 2.6)
    crate_count_range: tuple[int, int] = (0, 3)
    step_decoy_frac: float = 0.20
    decoy_frac: float = 0.35
    background_count_range: tuple[int, int] = (1, 3)
    cylinder_frac_pillar: float = 0.5
    cylinder_frac_other: float = 0.15
    flat_repeats: int = 6
    background_range_m: tuple[float, float] = (10.0, 22.0)


def tile_rng(seed: int, terrain_id: int, salt: int = 0) -> np.random.Generator:
    """A generator whose stream depends only on the tile, not on iteration order.

    Mirrors ``decorate.tile_rng``, with a salt so a tile's clutter draw is independent of its
    width arm and its walls -- otherwise turning one pass on would silently re-roll another.

    Args:
        seed: Base seed.
        terrain_id: The tile.
        salt: Per-pass constant.

    Returns:
        A seeded generator.
    """
    return np.random.default_rng(abs(hash((int(seed), int(terrain_id), int(salt)))) % (2**32))


def box(x: float, y: float, z0: float, sx: float, sy: float, sz: float,
        cls: str, kind: str = "box") -> dict:
    """One axis-aligned decor box, centred at ``(x, y)`` with its base at ``z0``.

    Args:
        x: Centre x in the tile frame (m).
        y: Centre y (m).
        z0: Base height (m).
        sx: Size along x (m).
        sy: Size along y (m).
        sz: Height (m).
        cls: One of :data:`NON_INTERACTABLE` or :data:`INTERACTABLE`.
        kind: ``"cylinder"`` for a round volume, else ``"box"``. A cylinder is a
            decoration-only kind in both simulators -- it exists for a depth silhouette a box
            cannot make, and never for a surface anything stands on. Its radius is inscribed
            in the smaller footprint extent, so a square footprint gives the obvious cylinder.

    Returns:
        An overlay dict in the manifest's schema, marked as decoration.
    """
    if cls not in NON_INTERACTABLE + INTERACTABLE:
        raise ValueError(f"unknown clutter class {cls!r}")
    return {"kind": kind, "walkable_top": False, "yaw": 0.0, "variant": "solid",
            "size_xyz": [float(sx), float(sy), float(sz)],
            "pos_xy": [float(x), float(y)], "base_z": float(z0),
            "decor": True, "clutter_class": cls}


def aabb(b: dict) -> np.ndarray:
    """``[x_min, y_min, x_max, y_max]`` of one overlay box, in its tile's frame."""
    (sx, sy, _sz), (px, py) = b["size_xyz"], b["pos_xy"]
    return np.array([px - sx / 2.0, py - sy / 2.0, px + sx / 2.0, py + sy / 2.0])


@dataclass(frozen=True)
class TileOccupancy:
    """Where the robot ACTUALLY went on one tile, from a closed-loop recording.

    Attributes:
        terrain_id: The tile.
        cell: Grid pitch (m).
        origin: Tile-frame ``(x, y)`` of cell ``(0, 0)``.
        grid: ``(nx, ny)`` bool, True where a recorded robot position fell. NOT dilated --
            the vetoes add their own padding, so one grid serves both tiers and the
            approach-zone radius does not have to be baked in.
        n_samples: Recorded positions that went into it, so a tile with too few to trust can
            be recognised rather than read as empty.
    """

    terrain_id: int
    cell: float
    origin: tuple[float, float]
    grid: np.ndarray
    n_samples: int


def occupancy_from_positions(terrain_id: int, xy: np.ndarray, size_x: float, size_y: float,
                             cell: float = 0.25, pad_m: float = 20.0) -> TileOccupancy:
    """Bin recorded tile-frame robot positions into an occupancy grid.

    Args:
        terrain_id: The tile.
        xy: ``(N, 2)`` recorded robot positions in the TILE's frame.
        size_x: Tile extent along x (m).
        size_y: Tile extent along y (m).
        cell: Grid pitch (m).
        pad_m: How far outside the tile the grid reaches, so clutter placed beyond the tile
            (background masses, a wall past the end of the traversal) is still vetted rather
            than falling off the array.

    Returns:
        The occupancy.
    """
    ox, oy = -float(pad_m), -float(pad_m)
    nx = int(np.ceil((float(size_x) + 2 * pad_m) / cell))
    ny = int(np.ceil((float(size_y) + 2 * pad_m) / cell))
    grid = np.zeros((nx, ny), dtype=bool)
    if len(xy):
        ix = np.clip(((xy[:, 0] - ox) / cell).astype(int), 0, nx - 1)
        iy = np.clip(((xy[:, 1] - oy) / cell).astype(int), 0, ny - 1)
        grid[ix, iy] = True
    return TileOccupancy(int(terrain_id), float(cell), (ox, oy), grid, int(len(xy)))


def clears_occupancy(b: dict, occ: TileOccupancy, margin: float = CLEARANCE_M) -> bool:
    """Whether a box misses every cell the robot was ever recorded in.

    Args:
        b: The overlay box.
        occ: The tile's recorded occupancy.
        margin: Extra clearance (m), on top of the body radius.

    Returns:
        True if no recorded position falls within the padded box.
    """
    x0, y0, x1, y1 = aabb(b)
    pad = BODY_RADIUS_M + float(margin)
    ox, oy = occ.origin
    i0 = max(0, int(np.floor((x0 - pad - ox) / occ.cell)))
    i1 = min(occ.grid.shape[0], int(np.ceil((x1 + pad - ox) / occ.cell)))
    j0 = max(0, int(np.floor((y0 - pad - oy) / occ.cell)))
    j1 = min(occ.grid.shape[1], int(np.ceil((y1 + pad - oy) / occ.cell)))
    if i0 >= i1 or j0 >= j1:
        return True
    return not occ.grid[i0:i1, j0:j1].any()


def approach_zone_clear(b: dict, occ: TileOccupancy,
                        zone: float = APPROACH_ZONE_M) -> bool:
    """Whether an interactable-looking decoy keeps its whole approach zone off the path.

    The zone is grown around the feature in every direction rather than only downrange: a
    tile's traversal axis is authored along +x, but the robot wanders laterally and a decoy
    abeam the path at half a metre is as much of a contradiction as one in front of it.

    Args:
        b: The overlay box.
        occ: The tile's recorded occupancy.
        zone: Approach-zone radius (m).

    Returns:
        True if the robot never enters the decoy's approach zone.
    """
    return clears_occupancy(b, occ, margin=CLEARANCE_M + float(zone))


def windows_that_penetrate(ref_xy: np.ndarray, boxes: list[dict],
                           margin: float = CLEARANCE_M) -> np.ndarray:
    """Which prediction windows have their reference pass through a clutter box.

    The generator predicts ~1.24 s ahead, so a wall four metres away is entirely correct to
    walk toward for the whole horizon; it only becomes a lie once the reference enters it
    within the window. Rather than forbid every box the motion eventually reaches -- which
    would make the close-range case unteachable -- the offending WINDOWS are dropped and the
    rest kept.

    Args:
        ref_xy: ``(W, F, 2)`` reference positions per window, in the tile frame.
        boxes: The tile's clutter.
        margin: Clearance added around each box (m).

    Returns:
        ``(W,)`` bool, True where that window's reference enters a box.
    """
    if not boxes or ref_xy.size == 0:
        return np.zeros(len(ref_xy), dtype=bool)
    pad = BODY_RADIUS_M + float(margin)
    hit = np.zeros(len(ref_xy), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = aabb(b)
        inside = ((ref_xy[..., 0] > x0 - pad) & (ref_xy[..., 0] < x1 + pad)
                  & (ref_xy[..., 1] > y0 - pad) & (ref_xy[..., 1] < y1 + pad))
        hit |= inside.any(axis=1)
    return hit


def distance_to_path(x: float, y: float, occ: TileOccupancy) -> float:
    """Distance from a point to the nearest cell the robot was recorded in (m)."""
    ij = np.argwhere(occ.grid)
    if not len(ij):
        return float("inf")
    ox, oy = occ.origin
    cx = ox + (ij[:, 0] + 0.5) * occ.cell
    cy = oy + (ij[:, 1] + 0.5) * occ.cell
    return float(np.min(np.hypot(cx - x, cy - y)))


def sample_in_band(occ: TileOccupancy, size_x: float, size_y: float, lo: float, hi: float,
                   rng: np.random.Generator, tries: int = 60) -> tuple[float, float] | None:
    """A position whose distance to the recorded path falls in ``[lo, hi]``, or ``None``.

    Placement is by DISTANCE BAND rather than uniformly over the tile, because uniform
    placement puts most of a variant where it cannot be seen: a tile is tens of metres across,
    the far camera resolves a few, so the boxes that matter are the ones a short way off the
    path and uniform sampling produces them only by accident. Measured on the first variant
    authored uniformly: median distance 11 m, and under 40% of boxes within camera reach.

    Rejection sampling rather than a distance transform, to keep this dependency-free; the
    bands are wide and a failure just means one fewer box.

    Args:
        occ: The tile's recorded occupancy.
        size_x: Tile extent along x (m).
        size_y: Tile extent along y (m).
        lo: Nearest acceptable distance to the path (m).
        hi: Furthest (m).
        rng: Draw source.
        tries: Attempts before giving up.

    Returns:
        ``(x, y)`` in the tile frame, or ``None`` if the band is unreachable on this tile.
    """
    for _ in range(tries):
        x = float(rng.uniform(-hi, size_x + hi))
        y = float(rng.uniform(-hi, size_y + hi))
        if lo <= distance_to_path(x, y, occ) <= hi:
            return x, y
    return None


def clutter_for_tile(tile: dict, occ: TileOccupancy, cfg: ClutterConfig) -> list[dict]:
    """Author one tile's clutter, vetted against its own motion.

    Placement is REJECTION-SAMPLED rather than solved: a candidate is drawn, checked against
    the veto for its tier, and dropped if it fails. A tile whose motion covers most of its
    area therefore ends up with little clutter, which is the right outcome -- there is
    nowhere safe to put any.

    Args:
        tile: The manifest tile (read for ``size_x`` / ``size_y`` / ``terrain_id``).
        occ: Where the robot actually went on this tile, from the recording.
        cfg: Knobs.

    Returns:
        Overlay dicts to append to the tile, all marked ``decor``.
    """
    sx, sy = float(tile["size_x"]), float(tile["size_y"])
    rng = tile_rng(cfg.seed, int(tile["terrain_id"]), salt=0xC17E)
    out: list[dict] = []

    def place(cls: str, band: tuple[float, float], sxx: float, syy: float, szz: float,
              z0: float = 0.0, round_frac: float = 0.0) -> None:
        """Draw a position in the band, build the volume, and accept it if the veto passes."""
        interactable = cls in INTERACTABLE
        pos = sample_in_band(occ, sx, sy, band[0], band[1], rng)
        if pos is None:
            return
        kind = "cylinder" if (round_frac and rng.random() < round_frac) else "box"
        b = box(pos[0], pos[1], z0, sxx, syy, szz, cls, kind=kind)
        ok = approach_zone_clear(b, occ) if interactable else clears_occupancy(b, occ)
        if ok:
            out.append(b)

    # Bands are in metres from the recorded path, and are the whole reason placement is not
    # uniform. The far camera resolves a few metres ahead (less than its nominal reach, since
    # the torso pitches nose-down while walking), so a box beyond that is scenery.
    if rng.random() < cfg.wall_frac:
        h = float(rng.uniform(*cfg.wall_size_range))
        place("wall", (1.5, 5.0), 0.15, _span(rng, min(3.0, sy), sy), h)
    for _ in range(int(rng.integers(*_incl(cfg.pillar_count_range)))):
        w = float(rng.uniform(0.15, 0.45))
        place("pillar", (1.0, 4.0), w, w, float(rng.uniform(1.5, 3.0)),
              round_frac=cfg.cylinder_frac_pillar)
    if rng.random() < cfg.overhang_frac:
        place("overhang", (1.0, 4.0), _span(rng, 0.4, 1.2), _span(rng, min(2.0, sy), sy),
              0.2, z0=float(rng.uniform(*cfg.overhang_z_range)))
    for _ in range(int(rng.integers(*_incl(cfg.crate_count_range)))):
        place("crate", (1.0, 5.0), float(rng.uniform(0.4, 1.0)), float(rng.uniform(0.4, 1.0)),
              float(rng.uniform(0.6, 1.2)), round_frac=cfg.cylinder_frac_other)
    if rng.random() < cfg.step_decoy_frac:
        place("step_decoy", (0.8, 3.0), float(rng.uniform(0.4, 1.2)),
              float(rng.uniform(0.4, 1.2)), float(rng.uniform(0.10, 0.30)))
    # A decoy must sit OUTSIDE its own approach zone, so it is drawn just beyond it rather
    # than uniformly and then rejected -- uniform draws survived the veto about 4% of the
    # time, which is too few of the one class the generator responds to most.
    if rng.random() < cfg.decoy_frac:
        cls = "decoy_box" if rng.random() < 0.5 else "decoy_stairs"
        lo = APPROACH_ZONE_M + BODY_RADIUS_M + 2 * CLEARANCE_M
        place(cls, (lo, lo + 3.0), float(rng.uniform(0.8, 1.4)), float(rng.uniform(0.8, 1.6)),
              float(rng.uniform(0.25, 0.5)))
    for _ in range(int(rng.integers(*_incl(cfg.background_count_range)))):
        place("background", cfg.background_range_m, float(rng.uniform(0.5, 3.0)),
              float(rng.uniform(0.5, 3.0)), float(rng.uniform(1.0, 4.0)),
              round_frac=cfg.cylinder_frac_other)
    return out


def _span(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A draw from ``[lo, hi]`` that survives a tile narrower than ``lo``.

    Tile widths vary by an order of magnitude across the library, so any width expressed as
    "at least N metres" has to be clamped rather than assumed -- the alternative is a crash
    on the narrow tiles, which is how this was found.
    """
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    return float(rng.uniform(lo, hi)) if hi > lo else lo


def _incl(rng_pair: tuple[int, int]) -> tuple[int, int]:
    """``numpy.random.Generator.integers`` is half-open; make a declared range inclusive."""
    return int(rng_pair[0]), int(rng_pair[1]) + 1


def clutter_for_flat(occ: TileOccupancy, size_x: float, size_y: float,
                      cfg: ClutterConfig, variant: int) -> list[dict]:
    """Author one FLAT-tile variant, in TILE-CENTRE-relative coordinates.

    Flat tiles are the reason this pass exists at all. They are 30% of training, they carry
    no manifest entry of their own, and ``decorate.py`` records that at the shipped 20 m
    buffer **0.0% of flat clips have a wall within camera reach** -- so on a third of the
    data the far camera sees a bare plane to the horizon, every episode. Terrain tiles at
    least have walls, even if the motion stops short of them.

    Two things differ from a terrain tile and both are easy to get wrong:

    * the output frame is the tile CENTRE, not its corner -- ``flat_tile_decor`` is resolved
      that way because a flat tile's size depends on ``flat_buffer``, and absolute
      coordinates would land somewhere else the moment it moved;
    * the veto is the POOLED occupancy of every flat clip, because flat tiles all share
      ``terrain_id = -1`` and any flat clip can be drawn onto any of them. Vetting against
      one tile's traffic would place clutter where a different clip walks.

    Args:
        occ: Pooled flat occupancy, in tile-CORNER coordinates.
        size_x: Flat tile extent along x (m).
        size_y: Along y (m).
        cfg: Knobs.
        variant: Which variant this is; seeds its own stream.

    Returns:
        Overlay dicts relative to the tile centre.
    """
    tile = {"terrain_id": -1, "size_x": size_x, "size_y": size_y}
    saved = cfg.seed
    boxes: list[dict] = []
    try:
        for rep in range(max(1, int(cfg.flat_repeats))):
            cfg.seed = saved * 100000 + int(variant) * 100 + rep
            boxes.extend(clutter_for_tile(tile, occ, cfg))
    finally:
        cfg.seed = saved
    half_x, half_y = float(size_x) / 2.0, float(size_y) / 2.0
    for b in boxes:
        b["pos_xy"] = [b["pos_xy"][0] - half_x, b["pos_xy"][1] - half_y]
    return boxes


def merge_flat_decor(existing: list[list[dict]], variants: list[list[dict]]) -> list[list[dict]]:
    """Combine a library's own flat-room variants with a clutter sidecar's, pairwise.

    EXTENDS rather than replaces. The rooms are not in the way -- at the shipped buffer they
    are out of camera range entirely -- so dropping them would remove geometry that costs
    nothing and occasionally shows, while keeping them costs nothing either.

    The result is as long as the longer of the two, so the round-robin over flat tiles keeps
    whatever period it had; a library with no rooms simply gets the clutter.

    Args:
        existing: The library's ``flat_tile_decor``, possibly empty.
        variants: The sidecar's flat variants.

    Returns:
        One overlay list per variant.
    """
    n = max(len(existing), len(variants)) if (existing or variants) else 0
    return [list(existing[i % len(existing)] if existing else [])
            + list(variants[i % len(variants)] if variants else [])
            for i in range(n)]


def merge_into_manifest(manifest: list[dict], doc: dict, source: str | None = None) -> int:
    """Append a clutter sidecar's boxes onto a manifest, IN PLACE. Returns boxes added.

    The sidecar is deliberately not a library: only the decoration changes between variants,
    and copying gigabytes of motion to vary one attribute would put the cost in the wrong
    place. What that buys in disk it owes in checking, because a sidecar can be paired with
    the wrong library and produce a plausible world in the wrong place -- so the source is
    verified rather than assumed.

    Args:
        manifest: The library's tile list; mutated.
        doc: The parsed sidecar.
        source: The library's filename, checked against the sidecar's if given.

    Returns:
        How many boxes were added.

    Raises:
        ValueError: On a schema this does not know, a source mismatch, or a terrain_id the
            manifest does not contain.
    """
    if int(doc.get("schema", 0)) != 1:
        raise ValueError(f"clutter sidecar schema {doc.get('schema')!r} is not 1")
    if source is not None and doc.get("source") and doc["source"] != source:
        raise ValueError(
            f"clutter sidecar was authored against {doc['source']!r}, not {source!r}. Its "
            "boxes are in TILE-LOCAL frames, so pairing it with another library places them "
            "on whatever tile happens to share an id.")
    by_id = {int(t["terrain_id"]): t for t in manifest if "terrain_id" in t}
    unknown = [k for k in doc["per_terrain_id"] if int(k) not in by_id]
    if unknown:
        raise ValueError(f"clutter names terrain_ids absent from the manifest: {unknown[:8]}")
    added = 0
    for k, boxes in doc["per_terrain_id"].items():
        by_id[int(k)].setdefault("overlays", []).extend(boxes)
        added += len(boxes)
    return added
