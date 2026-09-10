# Copyright 2026 Zachary Olkin. All rights reserved.

"""Free-form terrain: the shared foundation every fixed course is built on, plus three
DEPRECATED randomised section families.

**Read this first.  The module is two things and only one of them is live.**

LIVE -- and imported by every current course (:mod:`box2`, :mod:`step_courses`,
:mod:`varied_courses`) and by the RL reset, the goal termination and four builder scripts:
:func:`spawn_area`, :func:`_route` and :data:`ROUTE_FIELDS` / :data:`SPAWN_AREA_FIELDS`,
:data:`GENERATOR_TAG`, :data:`FREEFORM_SPAWN_JITTER_M`, :data:`COMMANDED_SPEED_RANGE_MPS`,
the goal geometry constants, :func:`walkable_ground_span` and :func:`validate_manifest`.
None of that is deprecated; it is the contract a manifest-only library satisfies.

DEPRECATED 2026-09-09 -- the randomised SECTION families (``flat`` / ``boxes`` / ``stairs``:
:data:`FAMILIES`, :class:`FreeformConfig`, :func:`make_flat_tile`, :func:`make_boxes_tile`,
:func:`make_stairs_tile`, :func:`generate_freeform_manifest`) and their builder
``scripts/rsl_rl/make_freeform_terrain.py``.  **Nothing has ever been fine-tuned on them.**
Every result on this branch came from the FIXED courses instead, because a fixed course can be
overfitted to, re-evaluated on and compared against MuJoCo, and a distribution cannot.  They
are kept, unused, for the reasoning below -- the anchor-free authoring, the angle-by-approach
rule and the apron trap are all still true statements about how to build terrain -- and
because :mod:`step_courses` takes its flight shape from the ``stairs`` family.  They declare
no spawn areas, so they are also the only families the eval's ground-only spawn rule cannot
be applied to.

Do not add a new course here.  Add it as a fixed course beside :mod:`box2`.

--------------------------------------------------------------------------------------------

Free-form terrain sections for the FM-driven fine-tune -- geometry with no clip behind it.

Every terrain the repo has shipped so far exists to MATCH a pre-made motion clip tile by tile:
the clip is baked in the tile's corner frame and the ground is authored under the recorded feet.
Under the fine-tune the reference is GENERATED perceptively (``reference_source="fm"``,
``history_from="generator"``), so that constraint is gone and terrain can be authored for its own
sake.  It is still not random -- the robot must not be steered into the side of a staircase --
so each tile here is one SECTION with a start, a goal and a straight centreline the steering
follows, declared in the tile's own ``route`` block.

Three families, one tile per generated INSTANCE (not one tile per family):

* ``flat``   -- one large walkable extent.
* ``boxes``  -- axis-aligned boxes down a corridor: meet one, jump on and off, or walk along a
  long one first.  **Boxes are met at an ANGLE by turning the APPROACH, never by rotating the
  box**: the section's route runs at ``route.start_yaw`` to the tile's own +x while every box
  stays axis-aligned, so the robot arrives at a box face off-square.  Rotating the box instead
  is refused upstream -- :class:`OverlayVolumeCfg` has no yaw and the RL loader hard-refuses a
  non-zero overlay ``yaw`` (it would be built unrotated in Isaac and rotated in MuJoCo).
* ``stairs`` -- straight flights: 5 platforms, 4 independently up/down flights, 0/1/2 collidable
  stairwell walls, normal or floating treads.

**Every tile is ``type == "terrain"``.**  A new type string would need coordinated edits in
``clip_terrain_loader``'s class split, ``clip_io._TERRAIN_TYPE_NAMES`` (append-only -- the codes
are stored per clip), ``eval/sampler.TERRAIN_CLASS_ORDER`` (raises on an unknown class) and both
MuJoCo emitters, for zero geometric gain.  The families are told apart by the existing per-tile
``family`` field, and ``generator == "freeform_v1"`` marks a tile as coming from here.

**The ground is an explicit walkable overlay and ``has_floor`` is False.**  The loader pools every
tile of a class to the class MAXIMUM footprint, so a ``has_floor=True`` 6x3 m section would be
rendered on a 46x10 m corner-anchored floor -- a constant offset the depth policy can key on, and
one that differs between Isaac (which pads to the class footprint) and MuJoCo (which builds its
floor from the tile's own size).  Authoring the ground removes the artefact and makes the two
simulators agree by construction.  The consequence a consumer must handle: the tile CELL is still
the pooled class footprint, so a cell centre is not on this section's ground.  Spawn from
``route.start_x`` / ``route.start_y``, never from the env origin.

**The route is ONE straight ray, and that is the runtime's model too.**  ``route`` carries
:data:`ROUTE_FIELDS` (the RL side's ``mdp.events.resets.FREEFORM_ROUTE_FIELDS``, pinned equal by
a test) in the tile-local corner frame -- the frame ``clip_frame_origin`` places geometry in, so
a route and the mesh it describes share one datum and cannot drift.  The
RL steering (``mdp...fm_tracking_cmd.freeform_yaw_rate``) projects the root onto that ray and
aims a fixed distance further along it, so the aim point RECEDES rather than being reached;
:func:`route_lookahead` here is its offline twin and deliberately does not clamp at the goal
either.  A polyline would be a second, richer statement of the same thing that the runtime could
not consume -- so there is none, and an angled approach is expressed by ``start_yaw``.

Everything is resolved OFFLINE and written into the manifest: the importer, both MuJoCo emitters
and both viewers rebuild the world from it and never need this module's RNG.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: Marks a tile as authored here.  Read by consumers that must tell a free-form section from a
#: clip-library tile, which carry no such key.
GENERATOR_TAG: str = "freeform_v1"

#: The three section families, in the order :func:`generate_freeform_manifest` lays them out.
FAMILIES: tuple[str, str, str] = ("flat", "boxes", "stairs")

# ---------------------------------------------------------------------------------------------
# Trained geometry, COPIED from transfer/sim/generate_stairs_scene.py
# ---------------------------------------------------------------------------------------------
# That file is the source of truth: it owns the MuJoCo playground emitter and every measurement
# behind these numbers (the shipped library's stair tiles, re-measured 2026-08-27).  It is not
# importable from here -- ``transfer/sim`` is not a package and is not on ``sys.path``, so
# importing it would mean a path hack with an import side effect inside a library module, and
# ``motion_gen`` stays dependency-light on purpose.  So they are copied, and
# ``tests/test_freeform_terrain.py`` loads the real file and fails if the two ever drift.

#: Riser heights (m) the shipped clip library's stair tiles were generated on.
TRAINED_RISERS: tuple[float, ...] = (0.13, 0.131, 0.15, 0.17)
#: Every trained tread (m), ascending and descending pooled.
TRAINED_TREADS: tuple[float, ...] = (0.178, 0.19, 0.209, 0.22, 0.221, 0.25, 0.28)
#: The ASCENDING half.  Up and down flights used different tread sets, so a flight drawing from
#: the pooled set would put a descending tread under an ascending flight -- a pair no tile has.
TRAINED_TREADS_UP: tuple[float, ...] = (0.209, 0.22, 0.221, 0.25, 0.28)
#: The DESCENDING half.  Stated in ``generate_stairs_scene``'s ``TRAINED_TREADS`` comment rather
#: than exported as its own constant; the test pins ``set(UP) | set(DOWN) == set(TRAINED_TREADS)``
#: so a change to either side of the split cannot land on one file only.
TRAINED_TREADS_DOWN: tuple[float, ...] = (0.178, 0.19, 0.22, 0.25)
#: Jump-on / jump-off box heights (m), a 0.05 grid over the library's measured 0.25-0.75 range.
TRAINED_BOX_HEIGHTS: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45,
                                          0.55, 0.60, 0.65, 0.70, 0.75)

# ---------------------------------------------------------------------------------------------
# Regulation stair geometry
# ---------------------------------------------------------------------------------------------
#: Blondel's rule: ``2 * riser + tread`` in metres.  This is what COUPLES the two dimensions.
#: Sampling riser and tread independently inside a code box produces pairs no real staircase has
#: (a 0.10 m riser with a 0.36 m tread is legal under IBC and exists nowhere), which would make
#: the "regulation" arm a statement about a rectangle rather than about real stairs.
BLONDEL_RANGE_M: tuple[float, float] = (0.610, 0.635)

#: Physical floor on a riser (m), AUTHORED -- neither IRC nor OSHA states a riser minimum.  Only
#: a guard: with the tread maxima below, Blondel already keeps every drawn riser above 0.13 m.
RISER_MIN_FLOOR_M: float = 0.075


@dataclass(frozen=True)
class StairCode:
    """One building code's stair box, as a sampling constraint.

    Attributes:
        name: Short identifier, recorded per flight in the manifest.
        riser_min: Minimum riser (m).  :data:`RISER_MIN_FLOOR_M` where the code states none.
        riser_max: Maximum riser (m).
        tread_min: Minimum tread depth (m).
        tread_max: Maximum tread depth (m).  **AUTHORED** -- codes set a tread MINIMUM and no
            maximum, so an upper bound has to come from somewhere; without one the Blondel draw
            would run the riser down to nothing.
        pitch_deg_range: Allowed ``atan(riser / tread)`` in degrees, or ``None`` where the code
            constrains only the two dimensions.
        cite: The clause these numbers come from.
    """

    name: str
    riser_min: float
    riser_max: float
    tread_min: float
    tread_max: float
    pitch_deg_range: tuple[float, float] | None
    cite: str


#: The three codes the regulation arm draws from.  Checked against the standards, and one earlier
#: claim was WRONG and is not restated: **the current OSHA standard has no riser MINIMUM.**
#:
#: Blondel makes two of the stated limits non-binding, which is worth knowing before anyone
#: "fixes" them: over the tread ranges below the drawn riser lands in
#: [0.127, 0.191] (IBC), [0.140, 0.191] (IRC) and [0.153, 0.197] (OSHA), so IBC's 4-inch riser
#: floor and IRC's 7.75-inch ceiling never bite.  OSHA's PITCH does bite -- a shallow draw at
#: 0.152 m / 0.305 m is 26.6 deg against a 30 deg floor -- which is why the sampler rejects.
STAIR_CODES: tuple[StairCode, ...] = (
    StairCode(name="ibc_egress", riser_min=0.1016, riser_max=0.1778,
              tread_min=0.2794, tread_max=0.3556, pitch_deg_range=None,
              cite="IBC 1011.5.2 (means of egress): riser 4-7 in, tread >= 11 in"),
    StairCode(name="irc_dwelling", riser_min=RISER_MIN_FLOOR_M, riser_max=0.1969,
              tread_min=0.2540, tread_max=0.3302, pitch_deg_range=None,
              cite="IBC 1011.5.2 exception / IRC R311.7.5 (dwellings): riser <= 7.75 in, "
                   "tread >= 10 in"),
    StairCode(name="osha_industrial", riser_min=RISER_MIN_FLOOR_M, riser_max=0.2413,
              tread_min=0.2413, tread_max=0.3048, pitch_deg_range=(30.0, 50.0),
              cite="OSHA 1910.25(c) (fixed industrial): pitch 30-50 deg, riser <= 9.5 in, "
                   "tread >= 9.5 in, width >= 22 in"),
)

#: Rejection budget for :func:`sample_regulation_step`.  Measured acceptance is 1.00 on both IBC
#: arms and 0.73 on the OSHA one, so exhausting this is a range-editing mistake, not bad luck --
#: hence the raise rather than a clamped fallback.
_BLONDEL_TRIES: int = 64

#: Which sampling arm a stair flight's ``(riser, tread)`` came from.  Recorded per flight so
#: success can be reported separately for in-band and deliberately-OOD geometry.
SAMPLING_ARMS: tuple[str, str] = ("regulation", "trained")

# ---------------------------------------------------------------------------------------------
# Shared section geometry
# ---------------------------------------------------------------------------------------------
#: Thickness (m) of the authored ground slab.  Only has to be thicker than any sensor cares
#: about; the top face is the walking surface and sits at z = 0 on every family.
GROUND_THICKNESS_M: float = 0.5

#: Breathing room (m) between the section's geometry and the tile edge, on all four sides.
#: Mirrors the box-clip generator's ``--footprint_margin`` role: the tile is the geometry's
#: bounding box plus this.
SECTION_MARGIN_M: float = 0.5

#: Wall thickness (m).  Matches ``ground_variants.WALL_T_M`` / the clutter pass, so a stairwell
#: wall and a decoration wall look alike to a camera.
WALL_T_M: float = 0.15

#: How far into the section the robot spawns / the goal sits, measured from the geometry's own
#: start and end (m).
SPAWN_INSET_M: float = 0.75

#: Half-extents of the ORDINARY single-start spawn jitter, in metres.  Defined here rather
#: than in the env cfg because two places need the same number and they must not drift: the
#: reset synthesises a fallback spawn area at these extents for a tile declaring none, and a
#: tile that DOES declare areas has to author its ground-level one at the same size or its
#: ground spawns stop being comparable to every run before it.  Measured cost of getting that
#: wrong: box2's ground third was authored 0.375 x 0.995 against this 0.25 x 0.25, a 4x
#: lateral spread that let the robot start a metre off the approach centreline -- and because
#: declared areas REPLACE the jitter, `pos_jitter_m` had silently stopped doing anything there.
FREEFORM_SPAWN_JITTER_M: tuple[float, float] = (0.25, 0.25)

#: The band a free-form training run samples continuously (m/s), stamped into
#: ``generator_config`` under the key ``clip_io.read_commanded_speed_range`` reads.
#:
#: These are the CLIP-FREE COMMAND TERM's own defaults (``FmTrackingCommandCfg``'s
#: ``freeform_speed_{min,max}``), restated here so the LIBRARY declares its band rather than
#: inheriting whatever the env cfg happens to say on the day it is recorded -- which is the
#: whole reason every other course stamps one.  `motion_gen` is IsaacLab-free and cannot
#: import that cfg to read them; ``test_speed_band_contract.py`` pins the pair equal.
COMMANDED_SPEED_RANGE_MPS: tuple[float, float] = (0.6, 2.6)
GOAL_INSET_M: float = 0.75

#: Slack (m) allowed in every geometric containment check: an overlay against its tile footprint,
#: and a goal band against the walkable ground it has to sit on.
_BOUNDS_TOL: float = 1e-6


@dataclass(frozen=True)
class FreeformConfig:
    """What to generate, and the ranges every draw comes from.

    Attributes:
        seed: Base seed.  A tile's geometry depends only on ``(seed, terrain_id)``.
        num_flat / num_boxes / num_stairs: Tiles per family.
        flat_size_x_range / flat_size_y_range: Walkable extent of a flat section (m).
        flat_min_route_len_m: The walk a flat section must still afford at its widest heading.
            The heading is capped so a route of this length fits within the tile's lateral room.
        flat_max_approach_deg: How far a flat section's route may run off the tile's +x.
            Non-zero on purpose: a section whose route is the tile axis never asks the yaw
            controller for anything, and turning IS in distribution on flat (the training
            library commands up to 0.80 rad/s there).
        box_max_approach_deg: How far a boxes section's route may run off the tile's +x.  This
            is the whole approach-angle mechanism for that family -- see :func:`make_boxes_tile`.
            Bounded rather than large: the route drifts ``L * tan(theta)`` laterally over an
            ``L``-metre section, and the tile has to contain that drift.
        corridor_width_range: Walkable width of a boxes section (m).
        boxes_per_tile_range: Inclusive box-count range per boxes section.
        box_long_prob: Probability a box is a LONG one (walk along the top, then jump off).
        box_short_depth_range / box_long_depth_range: Box x-extent (m) for the two roles.
        box_width_range: Box y-extent (m), before the corridor clamp.
        box_gap_range: Clear x-gap between consecutive boxes (m).
        box_side_clearance_m: Walkable width (m) that must remain beside a box.  See
            :func:`make_boxes_tile` for why bounding the box WIDTH by ``Wy - 2 * this`` is enough
            to guarantee it.
        box_lead_m / box_runout_m: Flat run-up before the first box / run-out past the last (m).
        stair_width_range: Flight width (m).  4 ft to 10 ft: the narrowest staircase found on a
            real building up to a wide outdoor flight.
        platform_len_range: Platform length (m).  Floored at the IBC 1011.6 landing minimum
            (depth >= the stair width, need not exceed 48 in); the short end is kept
            deliberately, because mjeval identifies a ~2 m landing as reading like a box top and
            that confusion is what the fine-tune exists to fix.
        steps_range: Inclusive RISER count per flight.
        stair_floating_prob: Probability a flight's treads are floating slabs rather than solid.
        stair_slab_thickness_range: Floating slab thickness (m); the shipped library's slabs
            measure 0.0305-0.0572.
        stair_wall_height_range: How far a stairwell wall stands above the section's HIGHEST
            walking surface (m).
        stair_approach_m / stair_runout_m: Apron length (m) before the first platform / past the
            last, at those platforms' own levels.
        max_structure_height_m: Cap on ``max(platform top) - min(platform top)``.  Four flights
            of 15 risers all climbing would reach 14 m, which is a three-storey stairwell rather
            than a terrain section.  **It is a cap on the DRAW, not on the geometry**:
            :func:`_cap_structure_height` shortens flights and bottoms out at one riser each, so
            a tile whose flights all run the same way has a floor of roughly the sum of its four
            risers -- measured over 960 tiles / 40 seeds, that floor reaches 0.720 m (mean
            0.380).  A cap under a tile's own floor is REFUSED at generation rather than
            silently exceeded; the shipped 4.0 never binds that way (0 refusals, 0 violations
            over those same 960 tiles).
    """

    seed: int = 0
    num_flat: int = 8
    num_boxes: int = 16
    num_stairs: int = 24

    flat_size_x_range: tuple[float, float] = (20.0, 40.0)
    flat_size_y_range: tuple[float, float] = (8.0, 16.0)
    flat_max_approach_deg: float = 20.0
    flat_min_route_len_m: float = 8.0

    corridor_width_range: tuple[float, float] = (3.0, 6.0)
    boxes_per_tile_range: tuple[int, int] = (2, 5)
    box_long_prob: float = 0.35
    box_short_depth_range: tuple[float, float] = (0.40, 0.90)
    box_long_depth_range: tuple[float, float] = (2.0, 5.0)
    box_width_range: tuple[float, float] = (0.90, 4.90)
    box_gap_range: tuple[float, float] = (1.5, 3.0)
    box_side_clearance_m: float = 0.6
    box_lead_m: float = 2.5
    box_runout_m: float = 2.5
    box_max_approach_deg: float = 15.0

    stair_width_range: tuple[float, float] = (1.219, 3.048)
    platform_len_range: tuple[float, float] = (1.219, 5.0)
    steps_range: tuple[int, int] = (1, 15)
    stair_floating_prob: float = 0.25
    stair_slab_thickness_range: tuple[float, float] = (0.0305, 0.0572)
    stair_wall_height_range: tuple[float, float] = (1.2, 2.4)
    stair_approach_m: float = 3.0
    stair_runout_m: float = 3.0
    max_structure_height_m: float = 4.0

    #: Populated by :meth:`__post_init__`; the id block each family occupies.
    _blocks: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Resolve the per-family ``terrain_id`` blocks once, so every lookup agrees."""
        counts = {"flat": int(self.num_flat), "boxes": int(self.num_boxes),
                  "stairs": int(self.num_stairs)}
        start = 0
        blocks: dict[str, tuple[int, int]] = {}
        for fam in FAMILIES:
            blocks[fam] = (start, start + counts[fam])
            start += counts[fam]
        object.__setattr__(self, "_blocks", blocks)

    @property
    def num_tiles(self) -> int:
        """Total tiles the config generates."""
        return int(self.num_flat) + int(self.num_boxes) + int(self.num_stairs)

    def family_of(self, terrain_id: int) -> str:
        """Which family a ``terrain_id`` belongs to.

        Args:
            terrain_id: The tile id.

        Returns:
            One of :data:`FAMILIES`.

        Raises:
            ValueError: If the id is outside every block.
        """
        for fam, (lo, hi) in self._blocks.items():
            if lo <= int(terrain_id) < hi:
                return fam
        raise ValueError(f"terrain_id {terrain_id} is outside every family block {self._blocks}")

    def index_in_family(self, terrain_id: int) -> int:
        """Position of a tile within its own family block (0-based)."""
        lo, _hi = self._blocks[self.family_of(terrain_id)]
        return int(terrain_id) - lo


# Salts keep a tile's sub-draws independent: a rejection inside one flight's Blondel loop must not
# move the next flight's geometry, or editing one code's range would silently re-roll the rest of
# the tile.  Same rationale as ``clutter.tile_rng``'s salt.
_SALT_TILE: int = 0
_SALT_FLIGHT: int = 0x10
_SALT_BOX: int = 0x20


def tile_rng(seed: int, terrain_id: int, salt: int = 0) -> np.random.Generator:
    """A generator whose stream depends only on the tile, not on iteration order.

    Mirrors ``decorate.tile_rng`` / ``clutter.tile_rng`` in role.  The three ints are handed to
    numpy's ``SeedSequence`` rather than folded together with shifts and XORs, because a fold has
    collisions -- ``(seed, terrain_id)`` pairs that XOR to the same value would generate the same
    tile -- and a seed sequence does not.

    Args:
        seed: Base seed.
        terrain_id: The tile.
        salt: Per-pass constant, so independent draws within one tile do not share a stream.

    Returns:
        A seeded generator.
    """
    return np.random.default_rng([int(seed), int(terrain_id) + 1, int(salt)])


# ---------------------------------------------------------------------------------------------
# Step sampling
# ---------------------------------------------------------------------------------------------
def sample_regulation_step(rng: np.random.Generator, code: StairCode) -> tuple[float, float]:
    """One ``(riser, tread)`` pair from a code box, COUPLED by Blondel's rule.

    Draws a tread inside the code's range and a Blondel sum, derives the riser from the two, and
    rejects the pair if it leaves the code's riser range or pitch band.  Rejection rather than
    clamping: a clamped pair satisfies the box but breaks the coupling, which is the one thing
    that makes this arm resemble real stairs.

    Args:
        rng: The flight's generator.
        code: The code box to draw inside.

    Returns:
        ``(riser, tread)`` in metres, uniform in the flight (IBC 1011.5.4 allows 3/8 in of
        variation within a flight; this emits none).

    Raises:
        ValueError: If :data:`_BLONDEL_TRIES` draws are all rejected, which means the code's
            ranges and the Blondel band no longer intersect -- an edit, not bad luck.
    """
    lo, hi = BLONDEL_RANGE_M
    for _ in range(_BLONDEL_TRIES):
        tread = float(rng.uniform(code.tread_min, code.tread_max))
        riser = 0.5 * (float(rng.uniform(lo, hi)) - tread)
        if not (code.riser_min <= riser <= code.riser_max):
            continue
        if code.pitch_deg_range is not None:
            pitch = math.degrees(math.atan2(riser, tread))
            if not (code.pitch_deg_range[0] <= pitch <= code.pitch_deg_range[1]):
                continue
        return riser, tread
    raise ValueError(
        f"stair code {code.name!r} rejected all {_BLONDEL_TRIES} Blondel draws: riser "
        f"[{code.riser_min}, {code.riser_max}], tread [{code.tread_min}, {code.tread_max}], "
        f"pitch {code.pitch_deg_range}, Blondel {BLONDEL_RANGE_M}. These no longer intersect."
    )


def sample_trained_step(rng: np.random.Generator, going_up: bool) -> tuple[float, float]:
    """One ``(riser, tread)`` pair off the grid the shipped clip library was generated on.

    Args:
        rng: The flight's generator.
        going_up: Ascending flights draw from :data:`TRAINED_TREADS_UP`, descending ones from
            :data:`TRAINED_TREADS_DOWN` -- the library used different sets, and a descending
            flight is steeper per tread than any ascending one the robot has seen.

    Returns:
        ``(riser, tread)`` in metres.
    """
    treads = TRAINED_TREADS_UP if going_up else TRAINED_TREADS_DOWN
    return float(rng.choice(TRAINED_RISERS)), float(rng.choice(treads))


# ---------------------------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------------------------
def _overlay(cx: float, cy: float, base_z: float, sx: float, sy: float, sz: float,
             *, kind: str = "box", walkable_top: bool = True, decor: bool = False,
             variant: str = "solid", slab_thickness: float = 0.1) -> dict:
    """One overlay dict in the manifest's own schema.

    ``yaw`` is written explicitly as ``0.0`` rather than omitted: the RL loader hard-refuses a
    non-zero one (it would be axis-aligned in Isaac and rotated in MuJoCo), and writing the field
    makes that a stated property of every box here rather than a default nobody looked at.

    Args:
        cx / cy: Footprint centre (m), tile-local corner frame.
        base_z: z of the bottom face (m).
        sx / sy / sz: Full extents (m).
        kind: ``"box"``, ``"obstacle"`` or ``"overhead"``; anything else is refused at load.
        walkable_top: Whether the top face is a landing surface.
        decor: Visual only (no collider, invisible to the height scan).
        variant: Support under the top face; ``"floating"`` is a thin slab with a gap beneath.
        slab_thickness: Floating slab thickness (m); ignored when solid.

    Returns:
        The overlay dict.
    """
    return {"kind": kind, "pos_xy": [float(cx), float(cy)],
            "size_xyz": [float(sx), float(sy), float(sz)], "base_z": float(base_z),
            "walkable_top": bool(walkable_top), "yaw": 0.0, "decor": bool(decor),
            "variant": str(variant), "slab_thickness": float(slab_thickness)}


def _ground_overlay(size_x: float, size_y: float) -> dict:
    """The section's own ground: one walkable slab spanning the tile, top face at z = 0.

    This is why every free-form tile carries ``has_floor=False``.  See the module docstring.

    Args:
        size_x / size_y: The tile footprint (m).

    Returns:
        The ground overlay dict.
    """
    return _overlay(size_x / 2.0, size_y / 2.0, -GROUND_THICKNESS_M,
                    size_x, size_y, GROUND_THICKNESS_M)


def _riser_slab(x0: float, x1: float, cy: float, width: float, top_z: float) -> dict | None:
    """A walkable slab from the ground up to ``top_z``, or ``None`` when it IS the ground.

    A platform whose top sits at z = 0 is the authored ground slab already; emitting a
    zero-height box for it would fail ``build_overlay``'s all-positive-extent check.

    Args:
        x0 / x1: The slab's x span (m).
        cy: Lateral centre (m).
        width: Lateral extent (m).
        top_z: Top face (m); ``<= 0`` returns ``None``.

    Returns:
        The overlay dict, or ``None``.
    """
    if top_z <= 0.0:
        return None
    return _overlay((x0 + x1) / 2.0, cy, 0.0, max(x1 - x0, 1e-3), width, top_z)


# ---------------------------------------------------------------------------------------------
# The route: one straight ray per section
# ---------------------------------------------------------------------------------------------
#: The fields every section's ``route`` block declares, in the TILE-LOCAL CORNER frame, and the
#: exact set the RL side reads (``mdp.events.resets.FREEFORM_ROUTE_FIELDS``).  Lengths in metres,
#: angles in radians; ``x`` is downrange within the tile's own ``frame_yaw`` and ``y`` lateral.
#:
#: * ``start_x`` / ``start_y`` -- where the robot spawns.
#: * ``start_z`` -- the WALKABLE SURFACE there, not a base height: the top of the start platform.
#:   0.0 on flat and boxes; on stairs it follows from the geometry rule that offsets the whole
#:   structure so its LOWEST platform top sits at z = 0.
#: * ``start_yaw`` -- the route's heading relative to the tile frame.  0.0 = straight down the
#:   section; non-zero is how the boxes family produces an angled approach to an axis-aligned box.
#: * ``goal_x`` -- the downrange goal line: past it, the section has been walked.
#: * ``goal_y`` / ``goal_half_width`` -- the goal band's lateral centre and half-extent.
#:
#: ``goal_z`` is emitted too, beyond this tuple: the walkable surface at the goal, which is the
#: last platform's top on a descending or climbing stairs section.  The RL table reads only the
#: seven, so the extra key costs it nothing.
ROUTE_FIELDS: tuple[str, ...] = (
    "start_x", "start_y", "start_z", "start_yaw", "goal_x", "goal_y", "goal_half_width",
)

#: Floor on a goal band's half-width (m).  The band is otherwise half the WALKABLE surface at the
#: goal, which on a 1.219 m flight is 0.61 m.  mjeval widened its goal to 2.5 m because nothing in
#: that stack steered and most timeouts were a robot that had cleared the terrain and passed
#: beside the disc; here the yaw controller runs, so a band matching the surface measures the
#: right thing -- "got onto the landing" rather than "ended up roughly over there".
#:
#: **The RL side CLAMPS UP to this number** (``mdp.freeform_goal_reached``'s ``min_half_width``,
#: which ``g1_finetune_env_cfg`` passes from here), so a declared band narrower than the floor is
#: not what the termination measures.  :func:`_fit_goal_half_width` therefore refuses a section
#: whose ground cannot hold the FLOOR, rather than declaring a band the runtime widens back off
#: the ground.
#:
#: **What the floor still buys, MEASURED, and it is not zero.**  Containment is checked against
#: :func:`walkable_ground_span`, which does not filter by HEIGHT (see its docstring for why: the
#: floor slab beside a raised runout is floor, and filtering would refuse every flight narrower
#: than ``2 * 0.75`` m).  So on a flight narrower than 1.5 m the clamp widens the band past the
#: flight's own edge onto the tile floor below it.  Swept over 1920 tiles / 40 seeds at the
#: shipped config: **113 tiles (5.9%), stairs ONLY, overhanging by at most 0.1395 m per side**
#: (median 0.0625 m, none over 0.15 m).  ``mdp.freeform_goal_reached`` tests only ``x >= goal_x``
#: and ``|y - goal_y|``, never z, so within that overhang a robot standing on the floor -- up to
#: 4 m below the goal, having never climbed -- scores the same successful truncation as one on
#: the runout.  It is bounded and it is narrow (the robot's pelvis must be within 0.14 m of the
#: structure's side, where a stairwell wall often sits), which is why it is recorded here rather
#: than closed: closing it means carrying ``goal_z`` into ``FREEFORM_ROUTE_FIELDS`` and choosing
#: a height tolerance, and neither is free.
GOAL_HALF_WIDTH_FLOOR_M: float = 0.75


def walkable_ground_span(overlays: Sequence[dict], x: float, y: float) -> tuple[float, float]:
    """The contiguous walkable-ground y-interval covering the tile-local point ``(x, y)``.

    ``has_floor`` is False on every free-form tile, so ground exists only where an overlay puts
    it: a y outside this interval is VOID, not floor.  Decoration and non-walkable volumes (the
    stairwell walls) are excluded -- neither is something to stand on -- and the intervals of
    everything else spanning ``x`` are merged, so a box sitting on the ground extends nothing.

    **Heights are deliberately not filtered.**  The question this answers is "is there floor
    here", which is what a goal band has to be contained by.  It is not a claim that every y in
    the interval is at the goal's own level: beside a raised stair runout the floor is the ground
    slab metres lower, which is still floor and still not void.

    Args:
        overlays: The tile's overlay list.
        x: Tile-local downrange coordinate (m) -- overlays that do not span it are ignored.
        y: Tile-local lateral coordinate (m); the returned interval is the one containing it.

    Returns:
        ``(y_lo, y_hi)`` of the walkable interval containing ``y`` (m).

    Raises:
        ValueError: If no walkable overlay covers the point at all, which means the route was
            laid out over a hole in the tile's own ground.
    """
    spans: list[tuple[float, float]] = []
    for ov in overlays:
        if bool(ov.get("decor", False)) or not bool(ov.get("walkable_top", False)):
            continue
        px, py = float(ov["pos_xy"][0]), float(ov["pos_xy"][1])
        ex, ey = float(ov["size_xyz"][0]), float(ov["size_xyz"][1])
        if px - ex / 2.0 - _BOUNDS_TOL <= x <= px + ex / 2.0 + _BOUNDS_TOL:
            spans.append((py - ey / 2.0, py + ey / 2.0))
    merged: list[list[float]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1] + _BOUNDS_TOL:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    for lo, hi in merged:
        if lo - _BOUNDS_TOL <= y <= hi + _BOUNDS_TOL:
            return lo, hi
    raise ValueError(f"no walkable overlay covers ({x}, {y}); the walkable spans at that x are "
                     f"{[tuple(m) for m in merged]}.")


def _fit_goal_half_width(desired: float, overlays: Sequence[dict],
                         goal_xy: tuple[float, float]) -> float:
    """A goal band's half-extent, cut to the walkable ground it is centred on.

    The band is SIZED by the section (half the corridor, half the flight, half the tile) but
    CENTRED on the goal -- and the goal is not on the tile's centreline, because an angled route
    offsets it laterally by ``length * sin(start_yaw)``.  Taking the size from the tile alone
    therefore covered void on one side and excluded real ground on the other, by that same
    amount: measured over the eight flat tiles of the default config at seed 0, 27.853 m of void
    covered and 27.853 m of ground excluded, every tile affected.  The consequence was not
    cosmetic -- ``mdp.freeform_goal_reached`` is wired ``time_out=True``, so a robot that had
    walked off the section and was falling scored a SUCCESSFUL truncation with its value
    bootstrapped.

    Cutting rather than re-centring: the termination measures ``|y - goal_y|``, so the goal is
    the band's centre by construction and the only free quantity is how wide it may be.

    Args:
        desired: The half-extent the family would use if the ground were unbounded (m).
        overlays: The tile's overlays, which is where its ground is (``has_floor`` is False).
        goal_xy: The goal band's centre, tile-local (m).

    Returns:
        ``min(desired, room)`` where ``room`` is the ground available either side of the goal.

    Raises:
        ValueError: If the ground either side is narrower than :data:`GOAL_HALF_WIDTH_FLOOR_M`.
            The RL side clamps the declared half-width UP to that floor, so a narrower band is
            not what gets measured -- it would be widened back over the void, silently.
    """
    lo, hi = walkable_ground_span(overlays, goal_xy[0], goal_xy[1])
    room = min(goal_xy[1] - lo, hi - goal_xy[1])
    if room < GOAL_HALF_WIDTH_FLOOR_M - _BOUNDS_TOL:
        raise ValueError(
            f"the goal at {tuple(goal_xy)} has only {room:.3f} m of walkable ground beside it, "
            f"under the {GOAL_HALF_WIDTH_FLOOR_M} m floor the RL termination clamps up to "
            f"(walkable span there: [{lo:.3f}, {hi:.3f}]). The declared band would be widened "
            "back over the void at runtime."
        )
    return min(float(desired), room)


#: Keys of one entry in a route's optional ``spawn_areas`` list.  A RECTANGLE the robot may be
#: placed anywhere inside, in the tile-local frame, plus the walkable surface height under it
#: and a relative draw weight.
#:
#: **Why areas rather than one start pose.** A section with a single start gives the generator
#: one initial condition per tile, so every episode meets the obstacle from the same place and
#: the whole set is degenerate in the variable a fine-tune is trying to broaden -- which is why
#: ``reset_on_freeform_section`` already jitters the single start. Areas generalise that: a
#: terrain can declare that standing ON an obstacle is a legal start, which is the only way to
#: collect the second half of a traverse without first succeeding at the first half. Measured
#: on the two-box course: every one of 352 successful episodes reached the goal by
#: stairs-down-ing off a box, because a correct two-jump traverse is rare enough that the
#: behaviour policy never produced one -- and AWR can only reweight what it is shown.
SPAWN_AREA_FIELDS: tuple[str, ...] = ("x", "y", "z", "half_x", "half_y", "yaw", "weight")


def spawn_area(x: float, y: float, z: float, half_x: float, half_y: float,
               yaw: float = 0.0, weight: float = 1.0) -> dict:
    """One rectangular spawn area, tile-local.

    Args:
        x: Rectangle centre, downrange (m).
        y: Rectangle centre, lateral (m).
        z: Walkable surface height under the rectangle (m).  For a box top this is the box's
            height, not 0 -- the reset adds the robot's standing height to it.
        half_x: Half-extent downrange (m).
        half_y: Half-extent lateral (m).
        yaw: Spawn heading relative to the tile frame (rad).
        weight: Relative draw weight.  Normalised across a tile's areas, so the numbers are
            ratios and adding an area does not silently re-weight the others' absolute rates.

    Returns:
        The area dict, carrying :data:`SPAWN_AREA_FIELDS`.
    """
    if half_x < 0.0 or half_y < 0.0:
        raise ValueError(f"spawn area half-extents must be >= 0; got ({half_x}, {half_y})")
    if weight <= 0.0:
        raise ValueError(f"spawn area weight must be > 0; got {weight}")
    return {"x": float(x), "y": float(y), "z": float(z), "half_x": float(half_x),
            "half_y": float(half_y), "yaw": float(yaw), "weight": float(weight)}


def _route(start_xy: tuple[float, float], start_z: float, start_yaw: float,
           goal_xy: tuple[float, float], goal_half_width: float, goal_z: float,
           spawn_areas: list[dict] | None = None) -> dict:
    """One section's route block, in the tile-local corner frame.

    Args:
        start_xy: Spawn position (m).
        start_z: Walkable surface under the spawn (m).
        start_yaw: Route heading relative to the tile frame (rad).
        goal_xy: Goal band centre (m).
        goal_half_width: Goal band half-extent (m).
        goal_z: Walkable surface at the goal (m).
        spawn_areas: Optional rectangles the robot may start inside, from :func:`spawn_area`.
            Omitted or empty means the single ``start_*`` pose, which is what every existing
            terrain declares and what the reset falls back to -- so this is additive and no
            library needs regenerating to keep working.

    Returns:
        The route dict, carrying :data:`ROUTE_FIELDS` plus ``goal_z`` and, when given,
        ``spawn_areas``.  The consuming table reads only :data:`ROUTE_FIELDS` and ignores
        extra keys, which is what lets a list ride inside a block of scalars.
    """
    out = {"start_x": float(start_xy[0]), "start_y": float(start_xy[1]),
           "start_z": float(start_z), "start_yaw": float(start_yaw),
           "goal_x": float(goal_xy[0]), "goal_y": float(goal_xy[1]),
           "goal_half_width": float(goal_half_width), "goal_z": float(goal_z)}
    if spawn_areas:
        out["spawn_areas"] = [dict(a) for a in spawn_areas]
    return out


def route_direction(route: dict) -> tuple[float, float]:
    """The route's unit direction in the tile frame.

    Args:
        route: A section's route block.

    Returns:
        ``(cos, sin)`` of ``start_yaw``.
    """
    return math.cos(route["start_yaw"]), math.sin(route["start_yaw"])


def route_length(route: dict) -> float:
    """Distance along the route from start to the goal line (m).

    Args:
        route: A section's route block.

    Returns:
        The start-to-goal distance.
    """
    return math.hypot(route["goal_x"] - route["start_x"], route["goal_y"] - route["start_y"])


def route_project(route: dict, xy: Sequence[float]) -> tuple[float, float]:
    """Where a point sits on the route: downrange distance and signed lateral offset.

    Args:
        route: A section's route block.
        xy: A tile-local ``(x, y)``.

    Returns:
        ``(along, lateral)`` in metres; ``lateral`` is positive to the route's left.  Neither is
        clamped -- the route is an infinite ray, so a robot behind the start reports a negative
        ``along`` rather than zero.
    """
    ux, uy = route_direction(route)
    dx = float(xy[0]) - route["start_x"]
    dy = float(xy[1]) - route["start_y"]
    return dx * ux + dy * uy, -dx * uy + dy * ux


def route_lookahead(route: dict, xy: Sequence[float],
                    lead_m: float) -> tuple[float, float]:
    """The point ``lead_m`` further along the route than the robot's own projection.

    This is the one call the steering makes.  **It does not clamp at the goal**, matching
    ``mdp...fm_tracking_cmd.freeform_yaw_rate``: the aim point recedes with the robot, which is
    what keeps the heading error meaningful right through the goal band instead of collapsing to
    a bearing at a point the robot is standing on.

    Args:
        route: A section's route block.
        xy: The robot's tile-local ``(x, y)``.
        lead_m: Look-ahead distance (m).

    Returns:
        The aim point ``(x, y)``, tile-local.
    """
    ux, uy = route_direction(route)
    along, _lateral = route_project(route, xy)
    s = along + float(lead_m)
    return route["start_x"] + s * ux, route["start_y"] + s * uy


# ---------------------------------------------------------------------------------------------
# Tile builders
# ---------------------------------------------------------------------------------------------
def _base_tile(terrain_id: int, family: str, size_x: float, size_y: float,
               overlays: list[dict], route: dict) -> dict:
    """The manifest entry every family shares, schema fields first.

    ``frame_yaw`` is 0.0 and STATED rather than omitted: it is the heading of the terrain's own
    frame, which the RL anchor mask resolves downrange / lateral in and the FM loss consumes as
    ``mask_frame_yaw_rel``.  Every section here runs along +x, so 0.0 is correct -- but a
    consumer must be able to read it rather than assume it.  An angled APPROACH is
    ``route.start_yaw``, which is a different quantity: it turns the robot within the tile's
    frame and leaves the anchor mask's axes where the geometry put them.

    Args:
        terrain_id: The tile id.
        family: One of :data:`FAMILIES`.
        size_x / size_y: Tile footprint (m).
        overlays: The tile's overlay volumes, ground first.
        route: The section's route block (see :func:`_route`).

    Returns:
        The manifest entry.
    """
    return {
        # --- the schema clip_terrain_loader parses ---
        "terrain_id": int(terrain_id),
        "type": "terrain",
        "size_x": float(size_x),
        "size_y": float(size_y),
        "frame_yaw": 0.0,
        "has_floor": False,
        "overlays": overlays,
        # --- additive runtime fields; consumers that predate them ignore them ---
        "generator": GENERATOR_TAG,
        "family": family,
        "route": route,
    }


def make_flat_tile(terrain_id: int, cfg: FreeformConfig) -> dict:
    """One flat section: a single large walkable extent.

    Authored as an explicit ground overlay like every other family, so the three cannot differ in
    how their ground reaches the two simulators.

    The route runs at an angle to the tile's +x.  A route along the tile axis would never ask the
    yaw controller for anything, and turning is in distribution on flat (the training library
    commands up to 0.80 rad/s on its flat and box families), so this is the family where the
    controller can be exercised without the wz extrapolation the stairs family forces.  The angle
    is bounded by the tile's own width, so the goal cannot leave the ground.

    **That angle is also why the goal BAND is fitted rather than taken from the tile.**  The goal
    is offset laterally by ``length * sin(start_yaw)``, so a half-width of ``size_y / 2`` -- sized
    from the tile but centred on the goal -- covered void on one side and excluded real ground on
    the other by exactly that offset (measured at seed 0: all 8 tiles, 27.853 m each way).  The
    band is now the widest one centred on the goal that stays on the ground, which over the
    shipped ranges is 1.250 m at the extreme angle (``SECTION_MARGIN_M + GOAL_INSET_M``, well
    over the RL floor of 0.75) and up to half the tile on a straight route.

    Args:
        terrain_id: The tile id.
        cfg: The generation config.

    Returns:
        The manifest entry.
    """
    rng = tile_rng(cfg.seed, terrain_id, _SALT_TILE)
    size_x = float(rng.uniform(*cfg.flat_size_x_range))
    size_y = float(rng.uniform(*cfg.flat_size_y_range))
    cy = size_y / 2.0
    start = (SPAWN_INSET_M + SECTION_MARGIN_M, cy)
    x_room = size_x - SECTION_MARGIN_M - GOAL_INSET_M - start[0]
    y_room = max(cy - SECTION_MARGIN_M - GOAL_INSET_M, 0.0)
    # The heading is drawn FIRST and the route is then cut to fit, rather than the goal being
    # pinned to the far edge and the heading squeezed to reach it. Pinning made the angle a
    # function of the tile's aspect ratio -- a 33 m section 12 m wide could only ever turn 3 deg,
    # which exercises nothing. Cutting the route instead keeps the angle and shortens the walk.
    yaw_cap = min(math.radians(cfg.flat_max_approach_deg),
                  math.asin(min(1.0, y_room / cfg.flat_min_route_len_m)))
    yaw = float(rng.uniform(-yaw_cap, yaw_cap))
    length = min(x_room / math.cos(yaw), y_room / max(abs(math.sin(yaw)), 1e-9))
    goal = (start[0] + length * math.cos(yaw), start[1] + length * math.sin(yaw))
    overlays = [_ground_overlay(size_x, size_y)]
    # Half the tile is what the band WANTS; the fit is what keeps it on the ground, because the
    # angled route puts the goal off the centreline. See :func:`_fit_goal_half_width`.
    route = _route(start, 0.0, yaw, goal,
                   _fit_goal_half_width(size_y / 2.0, overlays, goal), 0.0)
    return _base_tile(terrain_id, "flat", size_x, size_y, overlays, route)


def make_boxes_tile(terrain_id: int, cfg: FreeformConfig) -> dict:
    """One boxes section: axis-aligned boxes met at an angle down an angled route.

    **The approach angle comes from the ROUTE, never from the box.**  The route runs at
    ``start_yaw`` to the tile's +x while every box stays axis-aligned, so the robot arrives at a
    box face off-square by exactly that angle.  Rotating the box instead is refused upstream:
    :class:`OverlayVolumeCfg` has no yaw and ``_overlays_from_manifest`` raises on a non-zero
    overlay ``yaw``, because such a box is built unrotated in Isaac and rotated in MuJoCo.

    **Placement rule, in full.**  Boxes occupy DISJOINT x bands separated by a clear gap of at
    least ``min(box_gap_range)``, so no two can overlap whatever their lateral offsets are -- the
    non-overlap is a property of the layout, not something checked afterwards.  Each box centre
    sits on the route at its own x, plus a lateral offset drawn in TILE y; the offset PERPENDICULAR
    to the route is that times ``cos(start_yaw)``, so it is never larger, and it is what
    ``route_offset`` records.  The corridor is the band of half-width ``corridor_width / 2`` around
    the route; a box's y extent is bounded by ``corridor_width - 2 * box_side_clearance_m``, and
    since the two side clearances SUM to ``corridor_width - box_width``, that single bound
    guarantees at least ``box_side_clearance_m`` of walkable ground on one side of every box.  No
    box can seal the corridor.

    Roles: a SHORT box is met, jumped onto and off again; a LONG one (``box_long_prob``) is deep
    enough to walk along the top before jumping off.  Heights come off the trained grid, so the
    height axis is in distribution while the layout is not.

    Args:
        terrain_id: The tile id.
        cfg: The generation config.

    Returns:
        The manifest entry, carrying ``corridor_width`` and a ``boxes`` list of the resolved
        volumes (tile-local, after the y shift that puts the section's own extent on the tile).

    Raises:
        ValueError: If the corridor's low end cannot hold the narrowest box with clearance on
            both sides, which would make the seal-the-corridor bound unenforceable.
    """
    rng = tile_rng(cfg.seed, terrain_id, _SALT_TILE)
    corridor_w = float(rng.uniform(*cfg.corridor_width_range))
    yaw = float(rng.uniform(-1.0, 1.0)) * math.radians(cfg.box_max_approach_deg)
    slope = math.tan(yaw)
    n_boxes = int(rng.integers(cfg.boxes_per_tile_range[0], cfg.boxes_per_tile_range[1] + 1))
    width_cap = min(cfg.box_width_range[1], corridor_w - 2.0 * cfg.box_side_clearance_m)
    if width_cap < cfg.box_width_range[0]:
        raise ValueError(
            f"corridor_width_range low end {cfg.corridor_width_range[0]} leaves no room for a "
            f"{cfg.box_width_range[0]} m box with {cfg.box_side_clearance_m} m clearance a side."
        )

    # Built in a provisional frame whose origin is the SPAWN and whose x is the tile's x; the
    # route is the line y = slope * x through it. Shifted onto the tile at the end, once the
    # section's own y extent is known -- which it cannot be before the boxes are placed, because
    # the route drifts laterally as it runs.
    boxes: list[dict] = []
    x = cfg.box_lead_m
    for k in range(n_boxes):
        brng = tile_rng(cfg.seed, terrain_id, _SALT_BOX + k)
        long_box = bool(brng.random() < cfg.box_long_prob)
        depth = float(brng.uniform(*(cfg.box_long_depth_range if long_box
                                     else cfg.box_short_depth_range)))
        height = float(brng.choice(TRAINED_BOX_HEIGHTS))
        box_w = float(brng.uniform(cfg.box_width_range[0], width_cap))
        half_free = (corridor_w - box_w) / 2.0
        dy = float(brng.uniform(-half_free, half_free))
        cx = x + depth / 2.0
        boxes.append({"index": k, "role": "walk_along" if long_box else "jump",
                      "pos_xy": [cx, slope * cx + dy], "size_xyz": [depth, box_w, height],
                      "base_z": 0.0, "top_z": height, "x0": x, "x1": x + depth,
                      # Perpendicular to the route, which is the frame the corridor is defined
                      # in -- not the tile-y offset the draw was made in.
                      "route_offset": dy * math.cos(yaw)})
        x += depth
        if k < n_boxes - 1:
            x += float(brng.uniform(*cfg.box_gap_range))
    x_end = x + cfg.box_runout_m

    # The tile's y extent: the corridor band along the whole route, and every box footprint.
    route_y = (0.0, slope * x_end)
    y_lo = min(min(route_y) - corridor_w / 2.0,
               min(b["pos_xy"][1] - b["size_xyz"][1] / 2.0 for b in boxes))
    y_hi = max(max(route_y) + corridor_w / 2.0,
               max(b["pos_xy"][1] + b["size_xyz"][1] / 2.0 for b in boxes))
    # The spawn is inset a further ``SPAWN_INSET_M`` from the tile edge, and the goal keeps
    # ``GOAL_INSET_M`` of ground past it: the RL reset jitters the spawn by +-0.25 m and the goal
    # is a line the robot crosses, so neither may sit on the tile boundary.
    off_x, off_y = SECTION_MARGIN_M + SPAWN_INSET_M, SECTION_MARGIN_M - y_lo
    size_x = off_x + x_end + GOAL_INSET_M + SECTION_MARGIN_M
    size_y = (y_hi - y_lo) + 2.0 * SECTION_MARGIN_M
    for b in boxes:
        b["pos_xy"] = [b["pos_xy"][0] + off_x, b["pos_xy"][1] + off_y]
        b["x0"] += off_x
        b["x1"] += off_x

    overlays = [_ground_overlay(size_x, size_y)]
    overlays += [_overlay(b["pos_xy"][0], b["pos_xy"][1], 0.0, *b["size_xyz"]) for b in boxes]

    start = (off_x, off_y)
    goal = (x_end + off_x, slope * x_end + off_y)
    # The corridor's own half-width, fitted for the same reason as every other family -- it
    # already fits (the tile is the corridor band plus a margin), and the fit says so rather
    # than leaving it to be re-derived by a reader.
    route = _route(start, 0.0, yaw, goal,
                   _fit_goal_half_width(corridor_w / 2.0, overlays, goal), 0.0)
    tile = _base_tile(terrain_id, "boxes", size_x, size_y, overlays, route)
    tile["corridor_width"] = corridor_w
    tile["boxes"] = boxes
    return tile


def _draw_stair_flights(terrain_id: int, cfg: FreeformConfig, arm: str) -> list[dict]:
    """The four flights' direction, riser count and step dimensions, before the height cap.

    Args:
        terrain_id: The tile id.
        cfg: The generation config.
        arm: ``"regulation"`` or ``"trained"``.

    Returns:
        Four dicts carrying ``direction``, ``steps``, ``riser``, ``tread``, ``sampling``,
        ``code``, ``variant`` and ``slab_thickness``.
    """
    flights: list[dict] = []
    for i in range(4):
        frng = tile_rng(cfg.seed, terrain_id, _SALT_FLIGHT + i)
        going_up = bool(frng.random() < 0.5)
        steps = int(frng.integers(cfg.steps_range[0], cfg.steps_range[1] + 1))
        if arm == "regulation":
            code = STAIR_CODES[int(frng.integers(0, len(STAIR_CODES)))]
            riser, tread = sample_regulation_step(frng, code)
            code_name = code.name
        else:
            riser, tread = sample_trained_step(frng, going_up)
            code_name = None
        floating = bool(frng.random() < cfg.stair_floating_prob)
        slab = float(frng.uniform(*cfg.stair_slab_thickness_range))
        flights.append({"index": i, "direction": "up" if going_up else "down", "steps": steps,
                        "riser": riser, "tread": tread, "sampling": arm, "code": code_name,
                        "variant": "floating" if floating else "solid",
                        # Clamped to one riser so a slab can never be thicker than the step it
                        # tops -- mirrors ``box_clip.stair_tile_boxes``.  Never binds at the
                        # shipped ranges (0.031-0.057 against a riser of 0.13 or more).
                        "slab_thickness": min(slab, riser)})
    return flights


def _platform_levels(flights: list[dict]) -> list[float]:
    """Platform-top heights (m) relative to the first platform, before the z offset.

    Args:
        flights: The four flight dicts.

    Returns:
        Five levels, ``[0.0, ...]``.
    """
    levels = [0.0]
    for f in flights:
        step = float(f["steps"]) * float(f["riser"])
        levels.append(levels[-1] + (step if f["direction"] == "up" else -step))
    return levels


def _cap_structure_height(flights: list[dict], cap_m: float, terrain_id: int) -> None:
    """Shorten flights in place until the structure fits under ``cap_m``, or REFUSE the cap.

    Four flights of 15 risers all climbing reach 14 m, which is a three-storey stairwell rather
    than a terrain section.  The reduction is deterministic and uses no RNG -- take one riser off
    whichever flight has the most, ties to the lowest index -- so it cannot perturb any other
    draw in the tile.  Terminates because every pass removes a riser and the loop stops once no
    flight has more than one.

    **One riser per flight is the floor, so the cap is not always reachable**, and this used to
    return quietly when it was not: ``FreeformConfig(seed=3, num_flat=0, num_boxes=0,
    num_stairs=6, max_structure_height_m=0.5)`` generated and validated a tile spanning 0.714 m,
    43% over its own cap, with nothing said anywhere -- ``validate_manifest`` cannot catch it
    either, since the manifest does not carry the cap.  So it raises instead.  A tile's floor is
    set by the DIRECTIONS drawn (four one-riser flights all climbing sum; an up/down pair
    cancels), which is why the message quotes the achieved span rather than a config-wide bound.

    Args:
        flights: The four flight dicts, mutated.
        cap_m: Maximum ``max(level) - min(level)`` (m).
        terrain_id: The tile, for the refusal message.

    Raises:
        ValueError: If every flight is down to one riser and the span still exceeds ``cap_m``.
    """
    while True:
        levels = _platform_levels(flights)
        span = max(levels) - min(levels)
        if span <= cap_m:
            return
        tallest = max(range(len(flights)), key=lambda i: (flights[i]["steps"], -i))
        if flights[tallest]["steps"] <= 1:
            raise ValueError(
                f"tile {terrain_id}: max_structure_height_m={cap_m} is unreachable -- every "
                f"flight is down to one riser and the structure still spans {span:.3f} m "
                f"(directions {[f['direction'] for f in flights]}, risers "
                f"{[round(float(f['riser']), 4) for f in flights]}). The floor is the sum of "
                "the four one-riser flights, signed by their directions; raise the cap or "
                "narrow the riser ranges."
            )
        flights[tallest]["steps"] -= 1


def make_stairs_tile(terrain_id: int, cfg: FreeformConfig) -> dict:
    """One stairs section: 5 platforms, 4 independently up/down flights, 0-2 stairwell walls.

    **The whole structure is offset so its LOWEST platform top sits at z = 0**, and the ground
    slab's top face sits there too.  Flights go up AND down, so without the offset a section that
    descends first would put walking surfaces below the world floor, and an absolute-world-z fall
    termination would fire on a perfectly good descent.  Everything is authored upward from that
    datum, which is also why a platform at exactly z = 0 emits no box: it IS the ground.

    A flight of ``n`` risers changes level by ``n * riser`` and emits ``n - 1`` intermediate
    treads, the landing platform being the n-th surface.  Riser and tread are UNIFORM within a
    flight (IBC 1011.5.4 allows 3/8 in of variation; this emits none) and drawn per flight.

    **The sampling arm is a per-tile draw and is recorded on every flight.**  Half the stair tiles
    take ``(riser, tread)`` from the regulation box (Blondel-coupled -- see
    :func:`sample_regulation_step`) and half from the grid the shipped library was generated on,
    so success can be reported separately for in-band and deliberately-OOD geometry.  Alternating
    by the tile's index within the family rather than by a coin gives an exact split and still
    depends only on the tile id.

    **The stairwell walls are COLLIDABLE** (``decor=False``, ``walkable_top=False``), which is a
    deliberate departure and has consequences worth stating where they are authored:

    * the generator was FITTED on non-colliding decoration -- every wall in the training library
      is ``decor=True``, so the robot has never been pushed back on by one;
    * nothing in the MDP teaches obstacle avoidance.  ``undesired_contacts`` is **-0.5**
      against a ``clf_reward`` of 10, and its sensor list is thighs / torso / backpack /
      elbows -- no wrists and no hands, so an arm swept into a wall is not penalised at all.
      **-0.5 is the EFFECTIVE weight on the task that renders these walls**, and it is not the
      one declared beside the term: ``g1_clf_tracking_base`` declares -0.1 and
      ``G1TeacherEnvCfg.__post_init__`` then overwrites it with -0.5, which every terrain cfg
      down to ``G1FinetuneFreeformEnvCfg`` inherits (measured on a constructed cfg, not read
      off the declaration).  Quoting the declared -0.1 understates the penalty 5x and makes
      this bullet's case look stronger than it is;
    * a collidable wall IS visible to the height scan, where a decor one is not.  So this changes
      the privileged observation as well as the physics, and a scan-conditioned number measured
      with walls is not comparable to one measured without.

    Args:
        terrain_id: The tile id.
        cfg: The generation config.

    Returns:
        The manifest entry, carrying ``stair_platforms``, ``stair_flights``, ``stair_width``,
        ``stair_wall_sides`` and ``stair_sampling_arm``.
    """
    rng = tile_rng(cfg.seed, terrain_id, _SALT_TILE)
    arm = SAMPLING_ARMS[cfg.index_in_family(terrain_id) % len(SAMPLING_ARMS)]
    width = float(rng.uniform(*cfg.stair_width_range))
    wall_sides = [["left", "right"], ["left"], ["right"], []][int(rng.integers(0, 4))]
    plat_len = [float(rng.uniform(*cfg.platform_len_range)) for _ in range(5)]
    wall_clearance_above = float(rng.uniform(*cfg.stair_wall_height_range))

    flights = _draw_stair_flights(terrain_id, cfg, arm)
    _cap_structure_height(flights, cfg.max_structure_height_m, terrain_id)
    levels = _platform_levels(flights)
    z_shift = -min(levels)
    levels = [z + z_shift for z in levels]

    # The wall band is reserved whether or not a wall is drawn, so the footprint -- and therefore
    # the pooled class footprint every tile in the library is padded to -- does not depend on the
    # wall count.
    size_y = width + 2.0 * WALL_T_M + 2.0 * SECTION_MARGIN_M
    cy = size_y / 2.0

    overlays: list[dict] = []
    platforms: list[dict] = []
    x = SECTION_MARGIN_M
    struct_x0 = x
    approach = (x, x + cfg.stair_approach_m)
    x = approach[1]
    for i in range(5):
        platforms.append({"index": i, "x0": x, "x1": x + plat_len[i],
                          "length": plat_len[i], "top_z": levels[i]})
        x += plat_len[i]
        if i < 4:
            f = flights[i]
            run = (f["steps"] - 1) * f["tread"]
            f["x0"], f["x1"] = x, x + run
            f["z_from"], f["z_to"] = levels[i], levels[i + 1]
            x += run
    runout = (x, x + cfg.stair_runout_m)
    struct_x1 = runout[1]
    size_x = struct_x1 + SECTION_MARGIN_M

    overlays.append(_ground_overlay(size_x, size_y))
    for span, top in ((approach, levels[0]), (runout, levels[4])):
        slab = _riser_slab(span[0], span[1], cy, width, top)
        if slab is not None:
            overlays.append(slab)
    for p in platforms:
        slab = _riser_slab(p["x0"], p["x1"], cy, width, p["top_z"])
        if slab is not None:
            overlays.append(slab)
    for f in flights:
        overlays.extend(_flight_tread_overlays(f, cy, width))

    if wall_sides:
        wall_top = max(levels) + wall_clearance_above
        for side in wall_sides:
            sign = 1.0 if side == "left" else -1.0
            # Face clearance ZERO: the wall's inner face touches the flight edge, so the centre
            # sits half a thickness beyond it.  ``ground_variants.WALL_GAP_M`` is a CENTRE offset
            # and copying it here would bury half the wall inside the flight.
            overlays.append(_overlay((struct_x0 + struct_x1) / 2.0,
                                     cy + sign * (width / 2.0 + WALL_T_M / 2.0),
                                     0.0, struct_x1 - struct_x0, WALL_T_M, wall_top,
                                     kind="obstacle", walkable_top=False, decor=False))

    # start_yaw is 0 on stairs, deliberately: a staircase is straight and approaching one at an
    # angle is the failure this terrain exists to avoid, not a case to sample.  The goal band is
    # half the RUNOUT's own width, so "got down the stairs" means standing on the landing rather
    # than having ended up roughly past it.
    spawn = (approach[0] + SPAWN_INSET_M, cy)
    goal = (runout[1] - GOAL_INSET_M, cy)
    route = _route(spawn, levels[0], 0.0, goal,
                   _fit_goal_half_width(max(width / 2.0, GOAL_HALF_WIDTH_FLOOR_M),
                                        overlays, goal),
                   levels[4])
    tile = _base_tile(terrain_id, "stairs", size_x, size_y, overlays, route)
    tile["stair_width"] = width
    tile["stair_wall_sides"] = wall_sides
    tile["stair_wall_clearance_above_m"] = wall_clearance_above
    tile["stair_sampling_arm"] = arm
    tile["stair_approach"] = [approach[0], approach[1]]
    tile["stair_runout"] = [runout[0], runout[1]]
    tile["stair_platforms"] = platforms
    tile["stair_flights"] = flights
    return tile


def _flight_tread_overlays(flight: dict, cy: float, width: float) -> list[dict]:
    """The intermediate tread boxes of one flight (``steps - 1`` of them).

    Each tread is a solid pillar from the ground up to its own top, the convention
    ``box_clip.stair_tile_boxes`` uses, so the riser faces are real vertical surfaces with no gap
    for a foot to catch.  A FLOATING flight keeps ``pos`` / ``size`` identical and only adds
    ``variant`` + ``slab_thickness``, so the tread top, the AABB and every placement consumer see
    the same geometry as a solid staircase -- only the mesh below the tread differs.

    Args:
        flight: A flight dict, after :func:`make_stairs_tile` has filled ``x0`` / ``z_from``.
        cy: Lateral centre of the section (m).
        width: Flight width (m).

    Returns:
        The tread overlays; empty for a one-riser flight, which is a single step between two
        platforms and has no intermediate surface.
    """
    out: list[dict] = []
    sign = 1.0 if flight["direction"] == "up" else -1.0
    for k in range(1, int(flight["steps"])):
        top = flight["z_from"] + sign * k * flight["riser"]
        # A tread top lies strictly between two platform levels, both >= 0 and unequal, so it is
        # never at or below the ground -- which is what makes an unconditional box valid here.
        if top <= 0.0:
            raise ValueError(f"flight {flight['index']} tread {k} top {top} is not above ground; "
                             "the lowest-platform z offset was not applied.")
        x0 = flight["x0"] + (k - 1) * flight["tread"]
        out.append(_overlay(x0 + flight["tread"] / 2.0, cy, 0.0,
                            flight["tread"], width, top,
                            variant=flight["variant"], slab_thickness=flight["slab_thickness"]))
    return out


def generate_freeform_manifest(cfg: FreeformConfig) -> list[dict]:
    """The whole terrain manifest: one tile per instance, ids blocked by family.

    Ids run ``flat``, then ``boxes``, then ``stairs``.  A tile's geometry depends only on
    ``(cfg.seed, terrain_id)``, so regenerating with a different count for one family leaves
    every tile whose id did not move byte-identical.

    Args:
        cfg: The generation config.

    Returns:
        The manifest, sorted by ``terrain_id``.
    """
    builders = {"flat": make_flat_tile, "boxes": make_boxes_tile, "stairs": make_stairs_tile}
    return [builders[cfg.family_of(tid)](tid, cfg) for tid in range(cfg.num_tiles)]


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
#: The overlay kinds the RL loader accepts.  Copied from ``terrains/blocks/overlay.py``
#: (``OVERLAY_KINDS``), which cannot be imported here -- it pulls in IsaacLab.  Pinned by
#: ``tests/test_freeform_terrain.py`` against the real frozenset.
OVERLAY_KINDS: frozenset[str] = frozenset(("box", "obstacle", "overhead"))

def validate_manifest(manifest: list[dict],
                      families: Sequence[str] = FAMILIES) -> None:
    """Refuse a manifest the RL loader or a simulator would misread.

    Everything checked here is something that fails SILENTLY or far from its cause downstream:
    a duplicate id gets one tile and orphans the other, a non-zero overlay ``yaw`` is built
    unrotated in Isaac and rotated in MuJoCo, a zero extent raises inside ``build_overlay`` with
    no tile named, and geometry outside the footprint lands on a neighbouring tile once the
    layout packs them.  Run by the CLI before anything is written.

    Args:
        manifest: The tile list.
        families: The family names this manifest is allowed to carry.  Defaults to the three
            randomised free-form families; a FIXED course authored elsewhere (see
            :mod:`motion_gen.terrain.box2`) passes its own, because every other check here --
            ids, footprints, overlay yaw, the ground rule -- applies to it unchanged and
            forking the validator to gain one string would leave the real checks in two places.

    Raises:
        ValueError: Naming the tile and the field on the first violation.
    """
    ids = [int(m["terrain_id"]) for m in manifest]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate terrain_id(s) {dupes}: one tile is built, the rest orphaned.")
    for m in manifest:
        tid = int(m["terrain_id"])
        if m["type"] != "terrain":
            raise ValueError(f"tile {tid}: type {m['type']!r}; every free-form tile is 'terrain'.")
        if m.get("family") not in families:
            raise ValueError(f"tile {tid}: family {m.get('family')!r} not in {tuple(families)}.")
        if m.get("has_floor", True):
            raise ValueError(f"tile {tid}: has_floor must be False -- the section authors its "
                             "own ground overlay, so a loader floor would double it.")
        if abs(float(m.get("frame_yaw", 0.0))) > 1e-9:
            raise ValueError(f"tile {tid}: frame_yaw {m['frame_yaw']}; sections run along +x.")
        sx, sy = float(m["size_x"]), float(m["size_y"])
        for i, ov in enumerate(m["overlays"]):
            if str(ov["kind"]) not in OVERLAY_KINDS:
                raise ValueError(f"tile {tid} overlay {i}: kind {ov['kind']!r} not in "
                                 f"{sorted(OVERLAY_KINDS)}; an unknown kind is silently drawn "
                                 "as a box.")
            if abs(float(ov.get("yaw", 0.0))) > 1e-9:
                raise ValueError(f"tile {tid} overlay {i}: yaw {ov['yaw']}; the loader refuses a "
                                 "yawed overlay (axis-aligned in Isaac, rotated in MuJoCo).")
            ex, ey, ez = (float(v) for v in ov["size_xyz"])
            if not (ex > 0.0 and ey > 0.0 and ez > 0.0):
                raise ValueError(f"tile {tid} overlay {i}: size_xyz {ov['size_xyz']} is not "
                                 "all-positive.")
            px, py = float(ov["pos_xy"][0]), float(ov["pos_xy"][1])
            if (px - ex / 2.0 < -_BOUNDS_TOL or px + ex / 2.0 > sx + _BOUNDS_TOL
                    or py - ey / 2.0 < -_BOUNDS_TOL or py + ey / 2.0 > sy + _BOUNDS_TOL):
                raise ValueError(f"tile {tid} overlay {i}: footprint leaves the tile "
                                 f"({px}+-{ex / 2}, {py}+-{ey / 2}) vs ({sx}, {sy}).")
        _validate_route(m)
        if m["family"] == "stairs":
            _validate_stairs(m)


def _validate_route(tile: dict) -> None:
    """The route block, checked here rather than at the first reset.

    Every one of these fails far from its cause on the RL side: a missing field raises inside a
    table build hours into a run, a goal at or behind the start truncates every episode on its
    first step (which reads as a policy that never walks), and a start off the tile spawns the
    robot over the void between tiles.

    Args:
        tile: A manifest entry.

    Raises:
        ValueError: Naming the tile and the field.
    """
    tid = int(tile["terrain_id"])
    route = tile.get("route")
    if not route:
        raise ValueError(f"tile {tid}: no 'route' block; the RL side needs {list(ROUTE_FIELDS)}.")
    missing = [f for f in ROUTE_FIELDS if f not in route]
    if missing:
        raise ValueError(f"tile {tid}: route is missing {missing}.")
    sx, sy = float(tile["size_x"]), float(tile["size_y"])
    for xk, yk in (("start_x", "start_y"), ("goal_x", "goal_y")):
        x, y = float(route[xk]), float(route[yk])
        if not (0.0 <= x <= sx and 0.0 <= y <= sy):
            raise ValueError(f"tile {tid}: {xk}/{yk} ({x}, {y}) is outside the tile ({sx}, {sy}).")
    if route["goal_x"] <= route["start_x"]:
        raise ValueError(f"tile {tid}: goal_x {route['goal_x']} is at or behind start_x "
                         f"{route['start_x']}; every episode would truncate on its first step.")
    if route["goal_half_width"] <= 0.0:
        raise ValueError(f"tile {tid}: goal_half_width {route['goal_half_width']} is not positive.")
    # The goal must lie ON the route, or the steering aims down one line while the termination
    # watches another -- which reads as a policy that walks past its goal.
    _along, lateral = route_project(route, (route["goal_x"], route["goal_y"]))
    if abs(lateral) > 1e-6:
        raise ValueError(f"tile {tid}: the goal sits {lateral} m off its own route "
                         f"(start_yaw={route['start_yaw']}).")
    # The band is SIZED by the section and CENTRED on the goal, which an angled route moves off
    # the tile's centreline -- so containment is NOT implied by the overlay footprint check
    # above and is its own invariant. Checked against the EFFECTIVE band: the RL termination
    # clamps the declared half-width up to GOAL_HALF_WIDTH_FLOOR_M, so it is the widened band
    # that can cover void, and a validator reading only the declared one would pass a manifest
    # the runtime then gets wrong.
    gx, gy = float(route["goal_x"]), float(route["goal_y"])
    span_lo, span_hi = walkable_ground_span(tile["overlays"], gx, gy)
    half = max(float(route["goal_half_width"]), GOAL_HALF_WIDTH_FLOOR_M)
    if gy - half < span_lo - _BOUNDS_TOL or gy + half > span_hi + _BOUNDS_TOL:
        raise ValueError(
            f"tile {tid}: the goal band [{gy - half:.3f}, {gy + half:.3f}] leaves the walkable "
            f"ground [{span_lo:.3f}, {span_hi:.3f}] at goal_x={gx:.3f} (declared half-width "
            f"{route['goal_half_width']:.3f}, floored to {half:.3f}). The termination is wired "
            "time_out=True, so a robot over the void past the goal line scores a SUCCESSFUL "
            "truncation with its value bootstrapped."
        )


def _validate_stairs(tile: dict) -> None:
    """The stairs-only invariants: the z datum, the flight count and the wall clearance.

    Args:
        tile: A stairs manifest entry.

    Raises:
        ValueError: Naming the tile and the violated invariant.
    """
    tid = int(tile["terrain_id"])
    plats, flights = tile["stair_platforms"], tile["stair_flights"]
    if len(plats) != 5 or len(flights) != 4:
        raise ValueError(f"tile {tid}: {len(plats)} platforms / {len(flights)} flights; "
                         "the family is 5 platforms and 4 flights.")
    lowest = min(float(p["top_z"]) for p in plats)
    if abs(lowest) > 1e-9:
        raise ValueError(f"tile {tid}: lowest platform top is {lowest}, not 0. A descending "
                         "section below the world floor fires the absolute-z fall termination "
                         "on a perfectly good descent.")
    for f in flights:
        if f["sampling"] not in SAMPLING_ARMS:
            raise ValueError(f"tile {tid} flight {f['index']}: sampling {f['sampling']!r} not in "
                             f"{SAMPLING_ARMS}.")
        if not (1 <= int(f["steps"])):
            raise ValueError(f"tile {tid} flight {f['index']}: steps {f['steps']} < 1.")
    width, cy = float(tile["stair_width"]), float(tile["size_y"]) / 2.0
    for side in tile["stair_wall_sides"]:
        sign = 1.0 if side == "left" else -1.0
        want_face = cy + sign * width / 2.0
        walls = [o for o in tile["overlays"] if o["kind"] == "obstacle"]
        got = [o for o in walls
               if abs((o["pos_xy"][1] - sign * o["size_xyz"][1] / 2.0) - want_face) < 1e-6]
        if not got:
            raise ValueError(f"tile {tid}: the {side} wall's inner face does not touch the "
                             f"flight edge at y={want_face}. Face clearance is zero, so the "
                             "centre sits half a thickness beyond it.")
