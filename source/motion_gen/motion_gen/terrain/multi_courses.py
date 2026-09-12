# Copyright 2026 Zachary Olkin. All rights reserved.
"""The FOUR-type multi-terrain fine-tune library: boxes, up-into-down, entry stairs, in-dist.

``varied_courses`` builds two DRAWN families from one config. This adds the two that are not
drawn: the measured building entrance (:mod:`~motion_gen.terrain.entry_stairs`, in both
directions) and the ORIGINAL clip-library tiles, which are copied rather than generated.

**Why a separate module.** ``entry_stairs`` imports the shared goal geometry from
``varied_courses``, so a four-type generator living in ``varied_courses`` would close an import
cycle. This sits above both and imports each once.

**Why the in-distribution tiles are COPIED, never regenerated.** They are the terrain the
teacher trained on; their routes were authored once by ``scripts/rsl_rl/make_indist_routes.py``
against the source clip library, picking per tile the clip that starts furthest back and taking
that same clip's terminal pose as the goal. Re-deriving that here would be a second opinion
about what the in-distribution route IS, and the whole value of the control arm is that it is
the SAME question asked on the same ground. So the caller hands in the tiles from an existing
in-distribution library and this module samples and renumbers them, touching nothing else.

**The mix is the measurement, so the id blocks are the reduction key.** ``terrain_id`` is
assigned over the concatenation in a fixed family order, so each family occupies one contiguous
block and a report can reduce per type by id range. That matters more here than anywhere else:
a pooled success rate over a mix moves when the MIX moves, with no policy change at all, so a
multi-terrain arm is only readable against the single-terrain ones per type.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .entry_stairs import (
    ENTRY_STAIRS_DOWN_FAMILY,
    ENTRY_STAIRS_UP_FAMILY,
    make_entry_stairs_tile,
)
from .varied_courses import (
    BOXES_FAMILY,
    COMMANDED_SPEED_RANGE_MPS,
    UPDOWN_FAMILY,
    VariedCoursesCfg,
    goal_region_spec,
    make_boxes_tile,
    make_updown_tile,
)

#: Family order in the manifest, and therefore the ``terrain_id`` block order.
#:
#: Drawn families first, then the measured entrance, then the copied in-distribution tiles --
#: stated once here so the manifest, :func:`family_blocks` and any reader agree without
#: restating the order. The two entry-stairs directions are separate families; see
#: ``entry_stairs.ENTRY_STAIRS_UP_FAMILY`` for why they must not be pooled.
FAMILY_ORDER: tuple[str, ...] = (
    BOXES_FAMILY, UPDOWN_FAMILY, ENTRY_STAIRS_UP_FAMILY, ENTRY_STAIRS_DOWN_FAMILY,
)

#: Salt for the in-distribution sample, so it does not move when a drawn family's count does.
_INDIST_SALT: int = 90_001


@dataclass(frozen=True)
class MultiCounts:
    """Tile counts per type, as the fine-tune configs declare them.

    Attributes:
        boxes: Drawn box tiles.
        updown: Drawn up-into-down tiles.
        entry_stairs: Measured-entrance tiles, split between the two DIRECTIONS -- the
            ascending half rounded up, so an odd count gives the extra tile to the direction
            the course is named for.
        indist: In-distribution tiles, sampled from a library the caller supplies.
    """

    boxes: int = 0
    updown: int = 0
    entry_stairs: int = 0
    indist: int = 0

    def __post_init__(self) -> None:
        """Refuse negative counts, and a manifest with no tiles at all."""
        for name in ("boxes", "updown", "entry_stairs", "indist"):
            if getattr(self, name) < 0:
                raise ValueError(f"tile count {name} must be non-negative, "
                                 f"got {getattr(self, name)}.")
        if self.total == 0:
            raise ValueError("a multi-terrain library needs at least one tile.")

    @property
    def total(self) -> int:
        """Every tile the manifest will hold."""
        return self.boxes + self.updown + self.entry_stairs + self.indist

    @property
    def entry_up(self) -> int:
        """Ascending entry-stairs tiles."""
        return (self.entry_stairs + 1) // 2

    @property
    def entry_down(self) -> int:
        """Descending entry-stairs tiles."""
        return self.entry_stairs // 2


def tile_key(tile: dict) -> tuple[str, int]:
    """A copied tile's identity in its SOURCE library.

    ``terrain_id`` is renumbered on the way into a multi manifest, so it cannot identify a tile
    across libraries; ``source_terrain_id`` is what ``make_indist_routes.py`` stamps for exactly
    this. Paired with the family because the four source families number independently.

    Args:
        tile: A copied in-distribution tile.

    Returns:
        ``(family, source id)``.
    """
    return (str(tile.get("family", "?")),
            int(tile.get("source_terrain_id", tile.get("terrain_id", -1))))


def select_indist_tiles(source: list[dict], n: int, seed: int = 0,
                        exclude: set[tuple[str, int]] | None = None) -> list[dict]:
    """``n`` in-distribution tiles, balanced across the families the source holds.

    Balanced rather than uniform, because the source is not: the shipped in-distribution
    manifest is 15 / 20 / 20 / 20 across ``jump_off_front``, ``jump_on``, ``stairs_down`` and
    ``stairs_up``, so a uniform draw of 20 would return a different mix every seed and make the
    per-type breakdown's denominators wobble between libraries. Families are filled round-robin
    from a per-family shuffle, so a count that does not divide evenly spreads the remainder
    rather than starving the smallest family.

    Args:
        source: An in-distribution manifest's tiles, each carrying its own ``route``.
        n: How many to take. Must not exceed what the source holds, after ``exclude``.
        seed: Draw seed.
        exclude: Tiles already spent, as :func:`tile_key` results -- the training library's
            picks, when drawing a HOLDOUT. Without it the two draws are independent samples of
            one pool and overlap heavily: at 60 of 75 tiles they shared 80%, and
            ``jump_off_front`` (15 tiles, all of them taken by both) shared 100%, so a file
            named "holdout" reported mostly training tiles.

    Returns:
        Deep COPIES, in family order then source order, with ``terrain_id`` left alone --
        :func:`generate_multi_manifest` renumbers over the concatenation.

    Raises:
        ValueError: If the source is empty, carries a tile with no route, or holds fewer than
            ``n`` tiles.
    """
    if n <= 0:
        return []
    if not source:
        raise ValueError("no in-distribution tiles were supplied to sample from.")
    missing = [t.get("terrain_id") for t in source if not t.get("route")]
    if missing:
        raise ValueError(
            f"in-distribution tile(s) {missing} carry no route block. These tiles are COPIED "
            "rather than authored, so a missing route is a source-library problem: they come "
            "from make_indist_routes.py, which is what puts the route on them.")
    pool = [t for t in source if not exclude or tile_key(t) not in exclude]
    if n > len(pool):
        spent = len(source) - len(pool)
        raise ValueError(
            f"asked for {n} in-distribution tiles but only {len(pool)} of the source's "
            f"{len(source)} are available"
            + (f" ({spent} already spent on the training library). Lower the in-distribution "
               "tile count so a DISJOINT holdout fits, or build without one -- an overlapping "
               "holdout measures the tiles it was fitted on." if spent else "."))

    by_family: dict[str, list[dict]] = {}
    for t in pool:
        by_family.setdefault(str(t.get("family", "?")), []).append(t)

    rng = np.random.default_rng([int(seed), _INDIST_SALT])
    pools = {}
    for fam in sorted(by_family):
        tiles = by_family[fam]
        order = rng.permutation(len(tiles))
        pools[fam] = [tiles[i] for i in order]

    picked: list[dict] = []
    while len(picked) < n:
        took = False
        for fam in sorted(pools):
            if pools[fam] and len(picked) < n:
                picked.append(pools[fam].pop(0))
                took = True
        if not took:                      # unreachable: n <= len(source) is checked above
            break
    # Family order then source order, so the manifest is legible and reproducible.
    picked.sort(key=lambda t: (str(t.get("family", "?")), int(t.get("terrain_id", 0))))
    return [copy.deepcopy(t) for t in picked]


def generate_multi_manifest(counts: MultiCounts, cfg: VariedCoursesCfg | None = None,
                            indist_tiles: list[dict] | None = None,
                            exclude_indist: set[tuple[str, int]] | None = None) -> list[dict]:
    """One four-type manifest, ``terrain_id`` contiguous over the concatenation.

    Args:
        counts: Tiles per type.
        cfg: The drawn-course config; the defaults if omitted.
        indist_tiles: The in-distribution manifest to sample from. Required when
            ``counts.indist`` is non-zero.
        exclude_indist: Copied tiles already spent -- the training library's picks, when
            building a holdout. See :func:`select_indist_tiles`.

    Returns:
        The manifest, one entry per tile, in :data:`FAMILY_ORDER` then in-distribution order.

    Raises:
        ValueError: If in-distribution tiles are asked for and none are supplied.
    """
    cfg = cfg if cfg is not None else VariedCoursesCfg()
    if counts.indist and not indist_tiles:
        raise ValueError(
            f"counts.indist is {counts.indist} but no in-distribution tiles were supplied. "
            "They are COPIED from an existing library rather than generated -- pass the "
            "manifest of one built by scripts/rsl_rl/make_indist_routes.py.")

    tiles: list[dict] = []
    for i in range(counts.boxes):
        tiles.append(make_boxes_tile(len(tiles), cfg))
    for i in range(counts.updown):
        tiles.append(make_updown_tile(len(tiles), cfg))
    for i in range(counts.entry_up):
        tiles.append(make_entry_stairs_tile(len(tiles), cfg, descending=False))
    for i in range(counts.entry_down):
        tiles.append(make_entry_stairs_tile(len(tiles), cfg, descending=True))
    for t in select_indist_tiles(indist_tiles or [], counts.indist, int(cfg.seed),
                                 exclude=exclude_indist):
        t["terrain_id"] = len(tiles)
        tiles.append(t)
    return tiles


def family_blocks(manifest: list[dict]) -> dict[str, tuple[int, int]]:
    """The inclusive ``terrain_id`` range each family occupies, read off the manifest.

    The reduction key for the per-tile-type report. ``varied_courses.family_blocks`` derives
    its blocks from the two COUNTS instead, which is right there because the counts fully
    determine the layout. They do not here: the in-distribution block's internal split depends
    on which tiles the sample drew, so a count-derived version would have to re-run the draw --
    and would then silently disagree with the manifest at any seed but the one it assumed.

    Scanning is also the stronger check. Contiguity is a PROPERTY the report depends on rather
    than an assumption, so this verifies it instead of taking it on trust.

    Args:
        manifest: The tiles, as :func:`generate_multi_manifest` returned them.

    Returns:
        ``{family: (lo, hi)}``, in ascending id order.

    Raises:
        ValueError: If a family's tiles are not contiguous in ``terrain_id``, which would make
            an id-range reduction silently mix two families.
    """
    ids: dict[str, list[int]] = {}
    for t in manifest:
        ids.setdefault(str(t.get("family", "?")), []).append(int(t["terrain_id"]))
    out: dict[str, tuple[int, int]] = {}
    for fam, got in ids.items():
        got.sort()
        if got != list(range(got[0], got[-1] + 1)):
            raise ValueError(
                f"family {fam!r} occupies non-contiguous terrain ids {got}; a per-type report "
                "reduces by id RANGE, so it would pool this family with whatever sits in the "
                "gaps.")
        out[fam] = (got[0], got[-1])
    return dict(sorted(out.items(), key=lambda kv: kv[1][0]))


def validate_multi_manifest(manifest: list[dict]) -> None:
    """Check a four-type manifest, applying each half the checks that fit it.

    ``freeform.validate_manifest`` cannot be run over the whole thing, and the reason is not a
    technicality: it requires ``type == "terrain"``, which is the free-form AUTHORING schema,
    while the in-distribution tiles are original clip-library tiles carrying ``box`` and
    ``transition``. They are copied deliberately (see the module docstring) and
    ``make_indist_routes.py`` never ran that validator over them either.

    So the authored tiles get the full validator, the copied ones get the checks that apply to
    any tile, and the whole manifest gets the ones that only mean anything across the join --
    unique ids and contiguous family blocks, which is exactly what renumbering can break.

    Args:
        manifest: The tiles, as :func:`generate_multi_manifest` returned them.

    Raises:
        ValueError: Naming the tile and the field on the first violation.
    """
    from .freeform import validate_manifest  # noqa: PLC0415 -- avoids an import cycle at module load

    ids = [int(t["terrain_id"]) for t in manifest]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(
            f"duplicate terrain_id(s) {dupes}: one tile is built and the rest orphaned. This "
            "is what renumbering the copied in-distribution tiles gets wrong.")
    if ids != list(range(len(ids))):
        raise ValueError(
            f"terrain ids are not 0..{len(ids) - 1} in order; the report reduces per type by "
            "id RANGE, so a gap pools two families.")

    authored = [t for t in manifest if str(t.get("family", "")) in FAMILY_ORDER]
    if authored:
        validate_manifest(authored, families=FAMILY_ORDER)
    for t in manifest:
        if t not in authored:
            tid = int(t["terrain_id"])
            if not t.get("route"):
                raise ValueError(f"tile {tid}: copied in-distribution tile carries no route.")
            for f in ("size_x", "size_y", "overlays"):
                if not t.get(f):
                    raise ValueError(f"tile {tid}: copied tile is missing {f!r}.")
    family_blocks(manifest)      # raises if a family is not contiguous


def multi_goal_region_spec(counts: MultiCounts,
                           cfg: VariedCoursesCfg | None = None) -> dict:
    """The DECLARED ``goal_regions.json`` spec for a multi-terrain library.

    The drawn courses' spec unchanged -- every tile here, the entry stairs included, is BUILT
    to ``GOAL_HALF_WIDTH_M``, which is what lets one library-wide band be correct for all of
    them. Only the provenance differs, so a reader can tell which library a spec came from.

    **The height band is load-bearing on this library in a way it is not on the drawn ones.**
    There, every goal sits at ground level and the band only has to forgive a crouch. Here the
    ascending entry stairs and the in-distribution ``stairs_up`` / ``jump_on`` tiles finish a
    storey up, so the band is what separates "climbed it" from "walked past it on the floor" --
    ``make_indist_routes.py`` records exactly that recommendation per family. That only works
    now that ``goal_z`` reaches the runtime at all; it used to be dropped on the way, which put
    every goal's datum at the tile floor and made the band un-passable on precisely these tiles.

    Args:
        counts: The counts the manifest was built from.
        cfg: The drawn-course config; the defaults if omitted.

    Returns:
        A JSON-serialisable spec.
    """
    spec = goal_region_spec(counts.boxes, counts.updown, cfg)
    spec["provenance"]["source"] = "motion_gen.terrain.multi_courses"
    spec["provenance"]["counts"] = {
        "boxes": counts.boxes, "updown": counts.updown,
        "entry_stairs_up": counts.entry_up, "entry_stairs_down": counts.entry_down,
        "indist": counts.indist,
    }
    spec["provenance"]["note"] = (
        "DECLARED from the authored courses, not fitted: a manifest-only library has no clips "
        "to fit from. half_width_m is GOAL_HALF_WIDTH_M, which every tile of every family here "
        "is BUILT to. The height band matters on this library specifically: the ascending "
        "entry stairs and the in-distribution stairs_up / jump_on tiles finish a storey above "
        "their spawn, so it is what separates climbing the flight from walking past it on the "
        "floor beside it.")
    return spec


def multi_generator_config(counts: MultiCounts, cfg: VariedCoursesCfg | None = None,
                           indist_source: str | None = None,
                           commanded_speed_range_mps: tuple[float, float] | None = None) -> dict:
    """The ``generator_config`` block a multi-terrain library stamps.

    Carries :data:`COMMANDED_SPEED_RANGE_MPS` under the key
    ``clip_io.read_commanded_speed_range`` looks for, so the course owns its command band
    rather than it living in whichever launcher happened to pass an override.

    Args:
        counts: The counts the manifest was built from.
        cfg: The drawn-course config; the defaults if omitted.
        indist_source: Path of the library the in-distribution tiles were copied from,
            recorded so the control arm's provenance is in the file rather than in a shell
            history.
        commanded_speed_range_mps: The band to stamp. Defaults to
            :data:`~motion_gen.terrain.varied_courses.COMMANDED_SPEED_RANGE_MPS`, but a CALLER
            with its own declared band must pass it: the fine-tune configs declare one, and a
            library that stamped the module constant instead would silently ignore the number
            in the file. That is not hypothetical -- box2 and the curb shipped an
            ``intended_speeds_mps`` ladder for weeks that nothing read, so the whole declared
            band was a no-op and the loop used its own default.

    Returns:
        A JSON-serialisable config block.
    """
    cfg = cfg if cfg is not None else VariedCoursesCfg()
    band = (commanded_speed_range_mps if commanded_speed_range_mps is not None
            else COMMANDED_SPEED_RANGE_MPS)
    return {
        "seed": int(cfg.seed),
        "counts": {"flat": 0},
        "multi_courses": {
            "tiles": {"boxes": counts.boxes, "updown": counts.updown,
                      "entry_stairs_up": counts.entry_up,
                      "entry_stairs_down": counts.entry_down,
                      "indist": counts.indist},
            "indist_source": indist_source,
            "family_order": list(FAMILY_ORDER),
        },
        "freeform": {
            "section_counts": {"boxes": counts.boxes, "updown": counts.updown,
                               "entry_stairs": counts.entry_stairs,
                               "indist": counts.indist},
            "commanded_speed_range_mps": [float(v) for v in band],
        },
    }
