# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure-cfg companion to ``clip_terrain_importer.py``.

Kept separate so importing this cfg does NOT pull in the runtime importer (and its
``isaaclab.markers`` chain that pre-loads ``pxr`` before ``AppLauncher``). The
runtime class is resolved via the string ``class_type`` at ``cfg.class_type(cfg)``.
"""

from dataclasses import MISSING

from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass

from motion_gen.terrain.decorate import DEFAULT_FLAT_BUFFER


@configclass
class ClipTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for :class:`ClipTerrainImporter`.

    Builds a variable-size terrain directly from a clip library's manifest (one
    tile per terrain, per-class footprints, no uniform-grid padding) instead of a
    ``TerrainGeneratorCfg`` grid.  ``terrain_type`` / ``terrain_generator`` from the
    base cfg are ignored -- the importer assembles its own mesh from
    ``clip_library_path``.
    """

    class_type: type | str = "{DIR}.clip_terrain_importer:ClipTerrainImporter"

    clip_library_path: str = MISSING
    """Path to the clip-library ``.h5`` (terrain manifest + clips) to build from."""

    num_rows: int = 1
    """Replicas of the tile set (each a strip in y) -> more distinct env spawns."""

    render_decor: bool = True
    """Whether to build the library's visual-only decoration (walls, rooms, clutter) at all.

    ``False`` gives the terrain geometry alone, which is exactly the state an UNDECORATED
    library already produces -- ``create_decor_prim`` writes its unhittable placeholder, so
    ``/World/decor`` still exists and the depth cameras still have the target prim they are
    configured against. Nothing special-cases the off path.

    **This changes what the DEPTH CAMERAS see**, and therefore what the generator is
    conditioned on: the walls are the one channel that alters the image without altering what
    the robot stands on. So it is a VIEWING knob. Turning it off to look at geometry is fine;
    a measured run with it off is a different experiment from one with it on, and the two
    success rates are not comparable.

    The concrete reason it exists: walls ring each tile's own rendered FLOOR EDGE
    (``motion_gen.terrain.decorate.add_walls``) and are baked into the manifest, so where
    they sit is a property of the tile, not of how many tiles were loaded. In a small subset
    that puts a flat tile's room wall right where its NEIGHBOUR's terrain starts -- visually
    confusing without being wrong, and worth switching off while inspecting.
    """

    decor_overlay_path: str = ""
    """Optional clutter sidecar (``scripts/rsl_rl/author_clutter.py``) merged into the
    manifest at load.

    Visual-only geometry authored against a closed-loop recording, kept out of the library
    because the point is to have SEVERAL variants of it and only the decoration differs
    between them. Empty means the library as it stands, which is itself a variant -- the one
    every existing recording was made under.

    It cannot move the ground: everything it adds is ``decor``, which ``_tile_mesh`` splits
    out before the walkable mesh is built, so the tile layout and every recorded world pose
    stay valid. That is what lets a recording be re-rendered against it.

    TWO SCHEMAS arrive on this one path and they differ in exactly that respect: schema 1 is
    the clutter pass (visual-only boxes), schema 2 the terrain-variant pass, which EDITS the
    walkable ground. Pairing a schema-1 sidecar with ``render_decor=False`` is REFUSED -- every
    box it contributes would be dropped again, so the run would be identical to one with no
    sidecar while the log line claimed otherwise. Schema 2 is unaffected: its ground edits are
    not decoration."""

    render_decor: bool = True
    """Whether the VISUAL-ONLY decoration is built at all (``decor=True`` overlays: the tile
    walls, the flat-tile rooms, and anything a clutter sidecar adds).

    ``False`` drops them before the mesh is assembled. Nothing walkable moves: decoration lives
    on its own collider-less prim, the height scanner casts only ``/World/ground``, and the
    ground mesh, every tile corner and every clip placement come out byte-identical. Only a
    camera can tell the difference -- which is what the decor split is for.

    Off for the teacher's own TRAINING scene: nothing in this stack avoids obstacles, so walls
    and rooms teach the tracker nothing while costing render time and depth-camera pixels over
    a 12k-iteration run. ON wherever the generator is in the loop or being recorded -- the
    generator reads terrain from the cameras, and a vertical surface it has never seen is what
    it misreads as a box.

    KNOWN GAP: ``terrain_fingerprint`` hashes the MANIFEST, so a decor-on and a decor-off run
    are byte-identical in every exported artifact, ``tracker.onnx`` included. This flag is
    recorded only in the run's ``params/env.yaml`` and in the importer's own startup print,
    which is why the importer announces it. Do NOT fold it into the fingerprint: that hash
    answers "which GROUND was this trained on", and a decoration-only difference would then
    read as a ground difference to every check that compares it."""

    num_flat_tiles: int | None = None
    """Number of flat patches. ``None`` derives it from the library's generation
    metadata (``generator_config.counts.flat``) so flat/terrain/transition tile counts
    match the generation split -- NOT one tile per flat clip. See
    ``clip_terrain_loader.default_num_flat_tiles``."""

    col_gap: float = 1.0
    """Void gap (m) between tiles along x."""

    row_gap: float = 2.0
    """Void gap (m) between replica strips along y."""

    size_margin: float = 1.0
    """Extra flat (m) added to the terrain / transition footprints.

    Kept at 1.0 for now, and that is load-bearing rather than lazy: ``decorate.py`` resolves
    walls OFFLINE into absolute manifest positions against
    ``class_footprints(manifest, size_margin=...)``, and ``clips/g1_mb_v5_body_mode11_decor.h5``
    was decorated at 1.0. Raising this without re-running ``decorate_terrain.py`` puts every wall
    ~14 m inside the floor it is meant to bound -- and ``test_wall_footprint.py`` would NOT catch
    it, because it compares ``class_footprints`` against ``_class_fp`` (two constants) rather than
    against the library. See ``assemble_clip_terrain``'s wall-footprint guard.

    The case for a much larger value (fm-reference-window used 15.0 here and for
    :attr:`flat_buffer`): a rollout that drifts off its reference reaching the tile edge is a
    termination that looks exactly like a tracking failure, so a too-small margin silently
    converts "wandered" into "fell" in every survival number. 15 m covers ~5 s of drift at the
    library's top commanded speed. That argument is sound and unresolved -- but the sweep is
    confounded (changing the margin moves every env origin), so the prerequisite is a
    ``base_off_tile`` termination that makes off-mesh observable. TODO(tile-margin)."""

    flat_buffer: float = DEFAULT_FLAT_BUFFER
    """Flat-tile margin (m) on each side beyond the flat clips' bbox.

    20 m purely to be safe. The concern is that a small margin silently bounds the survival
    metric: the tracker task terminates only on ``time_out`` and ``base_fallen``, so a robot
    that walks off the platform registers as a fall and is indistinguishable from falling
    over. A wide margin makes that impossible; it is not a measured optimum.

    Sweeping 2 / 10 / 20 / 40 m produced no effect outside the binomial noise floor, in both
    closed loop and clip control, so nothing here is tuned -- 20 m just buys margin cheaply.

    The real fix is probably a divergence termination on the tracker (the teacher stops at
    0.3 m of base error; the tracker has no such term), so survival measures tracking rather
    than how long until the robot leaves the mesh."""

    transition_back_buffer: float = 1.0
    """Flat (m) prepended to each transition tile behind the robot's start."""
