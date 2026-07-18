# Copyright 2026 Zachary Olkin. All rights reserved.

"""Assemble a variable-size terrain from an offline clip library's manifest.

The offline clip generator embeds a ``terrain_manifest`` in the ``.h5`` describing
every terrain its clips were generated on.  Rather than force those terrains into
IsaacLab's uniform :class:`TerrainGeneratorCfg` grid (which requires one footprint
for every cell -> lots of flat padding when a bare 16 m staircase shares a grid
with a 42 m wandering-flat patch), this module builds the terrain **directly**:

* each terrain / transition manifest entry -> one tile, padded only to its CLASS
  footprint (terrain tiles share one size, transition tiles another, flat tiles a
  third -- see :func:`assemble_clip_terrain`);
* tiles are laid out in three per-class blocks, concatenated into one
  ``trimesh.Trimesh``, and handed to :meth:`TerrainImporter.import_mesh` with a
  matching ``env_origins`` tensor -- the same two public calls the grid path makes
  internally (see :class:`...terrains.clip_terrain_importer.ClipTerrainImporter`).

This avoids cross-class padding (no 42 m-wide staircase cells, no y-widened
transition tiles) at the cost of owning the placement loop here.

The offline clip is baked in the terrain's local-corner frame (``view_clip_viser``
overlays it on the mesh with no transform), so placing a clip on its tile is a
pure translation: x-corner on the tile corner, clip centerline on the tile
centerline (:func:`clip_frame_origin`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import trimesh

from .blocks import (
    CompositeSubTerrainCfg,
    FlatBlockCfg,
    OverlayVolumeCfg,
    SlopeBlockCfg,
    StairBlockCfg,
)
from .blocks.base import TerrainBlockCfg
from .blocks.composite import composite_terrain

_TRIVIAL_SKILL = "standing"          # blocks need a valid skill; unused for baked clips
_FLAT_TERRAIN_ID = -1
_PAD_TOL = 1e-4


@dataclass
class ClipTerrainLayout:
    """Per-tile layout tying the assembled mesh back to the clip terrains.

    Tiles are indexed ``[num_rows, num_tiles]`` (rows = replicas of the tile set
    for env capacity; a "tile" == one column).  Consumed by the importer to expose
    per-env terrain id + cell geometry.
    """

    num_rows: int
    num_tiles: int
    tile_terrain_id: torch.Tensor          # [num_rows, num_tiles] int64 (-1 = flat)
    tile_corner: torch.Tensor              # [num_rows, num_tiles, 3] world local-(0,0) of each tile
    tile_footprint: torch.Tensor           # [num_rows, num_tiles, 2] (fx, fy) per tile
    tile_clip_x_offset: torch.Tensor       # [num_rows, num_tiles] x-shift of the clip within the tile
    clip_size_y: dict[int, float]          # terrain id -> ORIGINAL corridor width (flat: footprint y)
    terrain_type_code: dict[int, int]      # terrain id -> {0 flat, 1 terrain, 2 transition, 3 slope, 4 box}
    # terrain id -> overlay volumes (boxes) in the tile's local-corner frame, as
    # emitted by ``composite_terrain``'s ``meta_data["overlays"]``.  The box world
    # (x, y) for a tile is ``tile_corner[:2] + overlay["pos_xy"]`` (Phase-3 reset).
    terrain_overlays: dict[int, list[dict]] = field(default_factory=dict)


def _blocks_from_manifest(entry: dict, footprint_x: float, front_pad: float = 0.0) -> list[TerrainBlockCfg]:
    """Rebuild one terrain's blocks (padded to ``footprint_x``) from its manifest.

    Mirrors ``terrain_clip_orchestrator.build_terrain``: an optional ``front_pad``
    flat block (buffer BEHIND the spawn), a ``FlatBlockCfg`` per flat band, a
    ``StairBlockCfg`` per stair band (uniform step dims, centered tread of width
    ``w``, no platforms), then a trailing flat pad so ``sum(size_x)`` hits the tile
    footprint.  The clip must be shifted forward by ``front_pad`` so its geometry
    sits after the buffer (see the ``tile_clip_x_offset`` handling in
    :func:`assemble_clip_terrain`).
    """
    dims = entry["dims"]
    w = float(dims["w"])
    blocks: list[TerrainBlockCfg] = []
    total = 0.0
    if front_pad > _PAD_TOL:
        blocks.append(FlatBlockCfg(size_x=front_pad, skill_probs={_TRIVIAL_SKILL: 1.0}))
        total += front_pad
    for band in entry["bands"]:
        length = float(band["aabb"][1]) - float(band["aabb"][0])
        total += length
        bt = band["block_type"]
        if bt == "StairBlock":
            h, d = float(dims["h"]), float(dims["d"])
            blocks.append(StairBlockCfg(
                size_x=length, direction="up",
                step_dim_options=[(h, d)], stair_width_range=(w, w),
                start_platform_length=0.0, end_platform_length=0.0,
                origin_distance_from_back=0.0, skill_probs={_TRIVIAL_SKILL: 1.0},
            ))
        elif bt == "SlopeBlock":
            # Uniform ramp: fixed total rise over this band's span (a rotated flat clip).
            rise = float(band.get("rise", dims.get("rise", 0.0)))
            blocks.append(SlopeBlockCfg(
                size_x=length, direction="up",
                rise_range=(rise, rise), slope_width_range=(w, w),
                start_platform_length=0.0, end_platform_length=0.0,
                skill_probs={_TRIVIAL_SKILL: 1.0},
            ))
        else:
            blocks.append(FlatBlockCfg(size_x=length, skill_probs={_TRIVIAL_SKILL: 1.0}))
    pad = footprint_x - total
    if pad > _PAD_TOL:
        blocks.append(FlatBlockCfg(size_x=pad, skill_probs={_TRIVIAL_SKILL: 1.0}))
    return blocks


def _flat_clip_travel(lib) -> tuple[float, float]:
    """Max (x-span, y-span) the flat clips' pelvis covers (0 if none).

    Sizes the flat tile so a flat clip (which samples lateral/yaw and wanders)
    never walks off its patch.
    """
    flat_mask = lib.terrain_id.to(torch.long) == _FLAT_TERRAIN_ID
    if not bool(flat_mask.any()) or lib.y_des.numel() == 0:
        return 0.0, 0.0
    names = lib.meta["pos_output_names"]
    px, py = names.index("pelvis_link:pos_x"), names.index("pelvis_link:pos_y")
    yf = lib.y_des[flat_mask]
    x_span = float((yf[:, :, px].amax(dim=1) - yf[:, :, px].amin(dim=1)).amax())
    y_span = float((yf[:, :, py].amax(dim=1) - yf[:, :, py].amin(dim=1)).amax())
    return x_span, y_span


def _overlays_from_manifest(entry: dict) -> list[OverlayVolumeCfg]:
    """Rebuild the overlay volumes (boxes) an ``entry`` carries (empty if none).

    Mirrors the ``overlays`` list the box-clip generator writes into the manifest
    (each ``{kind, pos_xy, size_xyz, base_z, walkable_top}`` in the clip's
    local-corner frame).
    """
    out: list[OverlayVolumeCfg] = []
    for ov in entry.get("overlays", []):
        out.append(OverlayVolumeCfg(
            kind=str(ov.get("kind", "box")),
            size_xyz=tuple(float(v) for v in ov["size_xyz"]),
            pos_xy=tuple(float(v) for v in ov["pos_xy"]),
            base_z=float(ov.get("base_z", 0.0)),
            walkable_top=bool(ov.get("walkable_top", True)),
        ))
    return out


def _tile_mesh(
    footprint: tuple[float, float],
    blocks: list[TerrainBlockCfg],
    overlays: list[OverlayVolumeCfg] | None = None,
) -> tuple[trimesh.Trimesh, list[dict]]:
    """Concatenated mesh of one tile (composite blocks + overlays) in local coords.

    Returns the mesh plus the per-overlay metadata list emitted by
    :func:`composite_terrain` (empty when the tile has no overlays).
    """
    sub = CompositeSubTerrainCfg(
        size=footprint, blocks=blocks, overlays=list(overlays or []),
        origin_block_index=0, start_z=0.0,
    )
    meshes, _origin, meta = composite_terrain(0.0, sub)
    return trimesh.util.concatenate(meshes), meta.get("overlays", [])


def default_num_flat_tiles(lib) -> int:
    """Number of flat tiles to build when the caller doesn't specify one.

    Derives from the generation metadata: the number of FLAT TERRAINS the library was
    generated with (``generator_config.counts.flat``), so the flat / terrain /
    transition tile counts match the requested generation split (total tiles ==
    ``num_terrains``).  Flat tiles are NOT one-per-clip -- each flat tile hosts many
    flat clips (the reset samples any flat clip onto a flat tile), so a library with
    thousands of flat clips still builds only ``counts.flat`` flat tiles.

    Falls back to the flat-clip count for pre-``generator_config`` libraries.

    Args:
        lib: A loaded :class:`ClipLibraryData`.

    Returns:
        The flat-tile count to build.
    """
    n_flat_clips = int((lib.terrain_id.to(torch.long) == _FLAT_TERRAIN_ID).sum())
    gen_counts = (lib.attrs.get("generator_config") or {}).get("counts") or {}
    return int(gen_counts.get("flat", n_flat_clips))


def assemble_clip_terrain(
    lib,
    *,
    num_rows: int = 1,
    num_flat_tiles: int | None = None,
    col_gap: float = 1.0,
    row_gap: float = 2.0,
    size_margin: float = 1.0,
    flat_buffer: float = 2.0,
    transition_back_buffer: float = 1.0,
) -> tuple[trimesh.Trimesh, torch.Tensor, ClipTerrainLayout]:
    """Build the whole terrain mesh + env origins from a clip library manifest.

    Each terrain / transition manifest entry becomes one tile padded to its CLASS
    footprint (terrain tiles one size, transition another); ``num_flat_tiles`` flat
    tiles (default = number of flat clips) are sized to the flat clips' travel plus
    ``flat_buffer`` on each side.  Tiles are packed into per-class near-square
    sub-grids stacked by class in y (replicas offset in x), concatenated, and
    centered at the world origin.

    Args:
        lib: A loaded :class:`ClipLibraryData` (``attrs["terrain_manifest"]`` +
            per-clip ``terrain_id`` + ``y_des`` for the flat-travel extent).
        num_rows: Replicas of the tile set (env capacity); each is a strip in y.
        num_flat_tiles: Number of flat patches (default: the library's flat-terrain
            count from ``generator_config.counts.flat`` via :func:`default_num_flat_tiles`,
            so flat/terrain/transition tile counts match the generation split).
        col_gap: Void gap (m) between tiles along x.
        row_gap: Void gap (m) between rows / class blocks along y.
        size_margin: Extra flat (m) added to the terrain / transition footprints.
        flat_buffer: Flat-tile margin (m) on EACH side beyond the flat clips' bbox
            (so a wandering flat clip doesn't reach the tile edge).
        transition_back_buffer: Flat (m) prepended to each transition tile BEHIND
            the robot's start (the clip is shifted forward by this much).

    Returns:
        ``(mesh, origins[num_rows, num_tiles, 3], layout)`` -- the concatenated
        terrain mesh, an env-origin tensor (tile centers, for
        :meth:`TerrainImporter.configure_env_origins`), and the
        :class:`ClipTerrainLayout`.

    Raises:
        ValueError: If the library has no ``terrain_manifest``.
    """
    manifest = lib.attrs.get("terrain_manifest")
    if not manifest:
        raise ValueError("clip library has no 'terrain_manifest'; not a terrain-aware library.")
    manifest = sorted(manifest, key=lambda m: int(m["terrain_id"]))
    terr = [m for m in manifest if m["type"] == "terrain"]
    trans = [m for m in manifest if m["type"] == "transition"]
    slope = [m for m in manifest if m["type"] == "slope"]
    box = [m for m in manifest if m["type"] == "box"]

    def _class_fp(ms: list[dict]) -> tuple[float, float]:
        if not ms:
            return (0.0, 0.0)
        return (max(float(m["size_x"]) for m in ms) + size_margin,
                max(float(m["size_y"]) for m in ms) + size_margin)

    terr_fp = _class_fp(terr)
    trans_fp0 = _class_fp(trans)
    # Transition tiles get a flat buffer BEHIND the spawn -> footprint x grows by it.
    trans_fp = (trans_fp0[0] + transition_back_buffer, trans_fp0[1])
    slope_fp = _class_fp(slope)
    box_fp = _class_fp(box)
    fx, fy = _flat_clip_travel(lib)
    flat_fp = (fx + 2.0 * flat_buffer, fy + 2.0 * flat_buffer)   # buffer on each side
    n_flat = default_num_flat_tiles(lib) if num_flat_tiles is None else int(num_flat_tiles)

    # Ordered tile specs: (terrain_id, type_code, footprint, blocks, clip_x_offset,
    # overlays).  ``overlays`` is the list of overlay volumes (boxes) for the tile,
    # empty for the surface-only terrain classes.
    clip_size_y: dict[int, float] = {}
    type_code: dict[int, int] = {}
    terrain_overlays: dict[int, list[dict]] = {}
    specs: list[tuple[int, int, tuple[float, float], list[TerrainBlockCfg], float, list[OverlayVolumeCfg]]] = []
    for m in terr:
        tid = int(m["terrain_id"])
        specs.append((tid, 1, terr_fp, _blocks_from_manifest(m, terr_fp[0]), 0.0, []))
        clip_size_y[tid], type_code[tid] = float(m["size_y"]), 1
    for m in trans:
        tid = int(m["terrain_id"])
        specs.append((tid, 2, trans_fp,
                      _blocks_from_manifest(m, trans_fp[0], front_pad=transition_back_buffer),
                      transition_back_buffer, []))
        clip_size_y[tid], type_code[tid] = float(m["size_y"]), 2
    for m in slope:
        tid = int(m["terrain_id"])
        specs.append((tid, 3, slope_fp, _blocks_from_manifest(m, slope_fp[0]), 0.0, []))
        clip_size_y[tid], type_code[tid] = float(m["size_y"]), 3
    for m in box:
        # A box tile is flat ground (padded to the box class footprint) carrying
        # standalone overlay volume(s) the clip jumps onto / off / over.
        tid = int(m["terrain_id"])
        ground = [FlatBlockCfg(size_x=box_fp[0], skill_probs={_TRIVIAL_SKILL: 1.0})]
        specs.append((tid, 4, box_fp, ground, 0.0, _overlays_from_manifest(m)))
        clip_size_y[tid], type_code[tid] = float(m["size_y"]), 4
    for _ in range(n_flat):
        specs.append((_FLAT_TERRAIN_ID, 0, flat_fp,
                      [FlatBlockCfg(size_x=flat_fp[0], skill_probs={_TRIVIAL_SKILL: 1.0})], 0.0, []))
    clip_size_y[_FLAT_TERRAIN_ID], type_code[_FLAT_TERRAIN_ID] = flat_fp[1], 0

    num_tiles = len(specs)
    if num_tiles == 0:
        raise ValueError("clip library manifest produced no tiles.")

    # Boxy layout: pack each class (a contiguous run of one ``type_code`` in
    # ``specs``) into a near-square sub-grid, and stack the class blocks in y.
    # Tiles keep their natural size; the empty space between differently-sized
    # tiles is void (no mesh), so this wastes no geometry -- it just makes the
    # field browsable instead of one long x-strip.
    base_xy: list[tuple[float, float]] = [(0.0, 0.0)] * num_tiles
    y_cursor = 0.0
    i = 0
    while i < num_tiles:
        tc = specs[i][1]
        j = i
        while j < num_tiles and specs[j][1] == tc:
            j += 1
        n = j - i
        fx_c, fy_c = specs[i][2]
        cols = max(1, math.ceil(math.sqrt(n)))
        for k in range(n):
            gr, gc = divmod(k, cols)
            base_xy[i + k] = (gc * (fx_c + col_gap), y_cursor + gr * (fy_c + row_gap))
        y_cursor += math.ceil(n / cols) * (fy_c + row_gap) + row_gap   # gap between class blocks
        i = j
    rep_w = max(base_xy[c][0] + specs[c][2][0] for c in range(num_tiles)) + 2.0 * row_gap

    meshes: list[trimesh.Trimesh] = []
    tile_corner = torch.zeros(num_rows, num_tiles, 3)
    tile_fp = torch.zeros(num_rows, num_tiles, 2)
    tile_tid = torch.zeros(num_rows, num_tiles, dtype=torch.long)
    tile_cxoff = torch.zeros(num_rows, num_tiles)
    for r in range(num_rows):
        xoff = r * rep_w                                    # replicas side-by-side in x
        for c, (tid, _tc, fp, blocks, cxoff, overlays) in enumerate(specs):
            x, y = xoff + base_xy[c][0], base_xy[c][1]
            m, ov_meta = _tile_mesh(fp, blocks, overlays)
            m.apply_translation((x, y, 0.0))
            meshes.append(m)
            tile_corner[r, c] = torch.tensor([x, y, 0.0])
            tile_fp[r, c] = torch.tensor(list(fp))
            tile_tid[r, c] = tid
            tile_cxoff[r, c] = cxoff
            if ov_meta:
                terrain_overlays[tid] = ov_meta

    mesh = trimesh.util.concatenate(meshes)
    # Center the whole layout at the world origin (xy only; keep z ground at 0).
    bmin, bmax = mesh.bounds
    cx, cy = 0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])
    mesh.apply_translation((-cx, -cy, 0.0))
    tile_corner[:, :, 0] -= cx
    tile_corner[:, :, 1] -= cy

    # Env origins = tile centers (the nominal spawn reference; clip placement uses
    # the corner/center directly, so this is just IsaacLab's env-origin bookkeeping).
    origins = tile_corner.clone()
    origins[:, :, 0] += tile_fp[:, :, 0] / 2.0
    origins[:, :, 1] += tile_fp[:, :, 1] / 2.0

    layout = ClipTerrainLayout(
        num_rows=num_rows, num_tiles=num_tiles, tile_terrain_id=tile_tid,
        tile_corner=tile_corner, tile_footprint=tile_fp, tile_clip_x_offset=tile_cxoff,
        clip_size_y=clip_size_y, terrain_type_code=type_code,
        terrain_overlays=terrain_overlays,
    )
    return mesh, origins, layout


def clip_frame_origin(
    cell_corner: torch.Tensor,
    cell_center: torch.Tensor,
    clip_y_center: torch.Tensor,
    clip_x_offset: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """World position to place a clip's local-corner (0,0,0) frame on its tile.

    The clip is baked in the terrain's local-corner frame, so aligning it to the
    tile is a pure translation (identity heading): x-corner on the tile corner
    (plus ``clip_x_offset`` -- the flat back-buffer prepended to transition tiles,
    so the clip's geometry sits AFTER the buffer), and the clip's centerline
    (``clip_y_center`` = ``size_y/2`` for terrain clips, ``~0`` for flat clips) on
    the tile centerline (absorbs the stair re-centering to the tile's footprint
    width).  z = tile ground (0); the clip's own baked z rides the treads with no
    double-count.

    Args:
        cell_corner: ``[N, 3]`` world position of each tile's local (0,0) corner.
        cell_center: ``[N, 3]`` world position of each tile's center.
        clip_y_center: ``[N]`` the clip frame y that lands on the tile centerline.
        clip_x_offset: ``[N]`` (or scalar) x-shift of the clip within the tile
            (the transition back-buffer; 0 for terrain / flat).

    Returns:
        ``[N, 3]`` world placement for each clip's local-corner frame.
    """
    out = torch.zeros_like(cell_corner)
    out[:, 0] = cell_corner[:, 0] + clip_x_offset
    out[:, 1] = cell_center[:, 1] - clip_y_center
    out[:, 2] = cell_corner[:, 2]
    return out
