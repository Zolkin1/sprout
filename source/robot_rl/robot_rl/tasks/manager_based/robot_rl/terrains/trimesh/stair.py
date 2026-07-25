# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

from motion_gen.motion_matching.stair_geom import stair_tread_layout
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_decor_cfg import (
    StairDecorCfg,
    StairSideDecorCfg,
)

if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_cfg import (
        MeshXStairsDownTerrainCfg,
        MeshXStairsUpTerrainCfg,
    )

_LAYOUT_TOL = 1e-5


def _make_project_fn(stair_top_centers: torch.Tensor, step_depth: float, num_steps: int):
    """Build the metadata projection closure for a stair sub-terrain.

    The closure maps a batch of (x, y) points in **local terrain corner
    coordinates** (x in [0, size_x], y in [0, size_y]) to the top-surface
    center of the stair the point sits over.

    Args:
        stair_top_centers: Tensor of shape (num_steps, 3). Per-stair top center
            (x_c, y_c, z_top) in local corner coordinates.
        step_depth: Step depth in x (m). Used for the integer-division lookup.
        num_steps: Number of stairs along x.

    Returns:
        Callable that takes a tensor of shape (N, 2) and returns a tensor of
        shape (N, 3).
    """

    def project(points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 2 or points.shape[-1] != 2:
            raise ValueError(
                f"project expects points of shape (N, 2); got {tuple(points.shape)}"
            )
        centers = stair_top_centers.to(points.device)
        x = points[..., 0]
        idx = torch.div(x, step_depth, rounding_mode="floor").long()
        idx = idx.clamp(0, num_steps - 1)
        return centers[idx]

    return project


def _resolve_decor(cfg) -> StairDecorCfg:
    """Return the explicit, deterministic decoration for a staircase span.

    If ``cfg.decor`` is set (the offline clip pipeline), it is used **verbatim** and
    no random numbers are drawn -- so the generator, the RL importer, and the viewers
    all rebuild identical geometry.  Otherwise the legacy probability fields
    (``float_prob`` / ``wall_prob`` / ``pole_prob`` + the thickness/height/spacing
    ranges) are sampled **once** into a :class:`StairDecorCfg`, now with independent
    per-side decoration (the old mutually-exclusive wall/pole coin flip is preserved
    for the legacy path, but expressed as resolved left/right sides).

    Args:
        cfg: A stair cfg (``StairBlockCfg`` / ``MeshXStairsUpTerrainCfg``) or any
            object exposing the decoration fields.

    Returns:
        The resolved :class:`StairDecorCfg`.
    """
    existing = getattr(cfg, "decor", None)
    if existing is not None:
        return existing

    float_prob = float(getattr(cfg, "float_prob", 0.0))
    is_floating = np.random.random() < float_prob
    float_thickness = (
        float(np.random.uniform(*getattr(cfg, "float_thick_range", (0.025, 0.075))))
        if is_floating else 0.0
    )

    def _side(kind: str) -> StairSideDecorCfg:
        if kind == "wall":
            return StairSideDecorCfg(
                kind="wall",
                thickness=float(getattr(cfg, "wall_thickness", 0.1)),
                height=float(getattr(cfg, "wall_height", 1.5)),
            )
        if kind == "posts":
            return StairSideDecorCfg(
                kind="posts",
                thickness=float(np.random.uniform(*getattr(cfg, "pole_thickness_range", (0.05, 0.15)))),
                height=float(getattr(cfg, "pole_height", 1.5)),
                spacing=float(np.random.uniform(*getattr(cfg, "pole_spacing_range", (0.3, 0.8)))),
            )
        return StairSideDecorCfg(kind="none")

    left_kind = right_kind = "none"
    if np.random.random() < float(getattr(cfg, "wall_prob", 0.0)):
        kind = "posts" if np.random.random() < float(getattr(cfg, "pole_prob", 0.0)) else "wall"
        if np.random.random() < 0.5:            # both sides
            left_kind = right_kind = kind
        elif np.random.random() < 0.5:          # one random side
            left_kind = kind
        else:
            right_kind = kind

    return StairDecorCfg(
        float_thickness=float_thickness,
        left=_side(left_kind),
        right=_side(right_kind),
    )


def _add_wall_one_side(
    meshes: list[trimesh.Trimesh],
    side: StairSideDecorCfg,
    *,
    left: bool,
    size_x: float,
    x_offset: float,
    y_center: float,
    stair_width: float,
    z_max: float,
    z_min: float,
    step_height: float,
) -> None:
    """Append one solid full-length sidewall box to ``meshes`` (deterministic)."""
    wall_z_bot = z_min - step_height
    wall_z_top = z_max + float(side.height)
    wall_h = wall_z_top - wall_z_bot
    wall_cz = (wall_z_top + wall_z_bot) / 2.0
    sgn = -1.0 if left else 1.0
    cy = y_center + sgn * (stair_width / 2.0 + float(side.thickness) / 2.0)
    dims = (size_x, float(side.thickness), wall_h)
    pos = (x_offset + size_x / 2.0, cy, wall_cz)
    meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))


def _add_posts_one_side(
    meshes: list[trimesh.Trimesh],
    side: StairSideDecorCfg,
    *,
    left: bool,
    size_x: float,
    x_offset: float,
    y_center: float,
    stair_width: float,
    z_max: float,
    z_min: float,
    step_height: float,
) -> None:
    """March uniform vertical railing posts along one stair edge (deterministic)."""
    thickness = float(side.thickness)
    spacing = float(side.spacing)
    if thickness <= _LAYOUT_TOL:
        return
    pole_z_bot = z_min - step_height
    pole_z_top = z_max + float(side.height)
    pole_h = pole_z_top - pole_z_bot
    pole_cz = (pole_z_top + pole_z_bot) / 2.0
    sgn = -1.0 if left else 1.0
    cy = y_center + sgn * (stair_width / 2.0 + thickness / 2.0)
    current_x = 0.0
    while current_x + thickness <= size_x + _LAYOUT_TOL:
        cx = x_offset + current_x + thickness / 2.0
        dims = (thickness, thickness, pole_h)
        pos = (cx, cy, pole_cz)
        meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))
        current_x += thickness + spacing


def _rounded_nosing(
    front_x: float,
    y_center: float,
    top_z: float,
    width: float,
    radius: float,
    segments: int,
    step_height: float,
    step_depth: float,
) -> trimesh.Trimesh | None:
    """Build a rounded (bull-nose) lip along a tread's leading top edge.

    A cylinder (axis along y, spanning the tread width) sits with its top flush at
    ``top_z`` and its axis over the tread's leading edge ``front_x``, so it rounds the
    top-front corner (protruding forward of the riser by ``radius`` and blending back
    into the tread).  No CSG -- the cylinder simply overlaps the tread box.  The
    radius is clamped to ``<= min(step_height, step_depth) / 2`` so the lip stays
    within the riser and never reaches the neighbouring tread.

    Returns ``None`` if the clamped radius is negligible.
    """
    r = min(float(radius), 0.5 * min(float(step_height), float(step_depth)))
    if r <= _LAYOUT_TOL:
        return None
    sections = max(6, 2 * int(segments))
    cyl = trimesh.creation.cylinder(radius=r, height=float(width), sections=sections)
    # trimesh cylinders are built along +z; rotate so the axis runs along y.
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, (1.0, 0.0, 0.0)))
    cyl.apply_translation((float(front_x), float(y_center), float(top_z) - r))
    return cyl


def _apply_decor(
    meshes: list[trimesh.Trimesh],
    decor: StairDecorCfg,
    *,
    size_x: float,
    x_offset: float,
    y_center: float,
    stair_width: float,
    z_max: float,
    z_min: float,
    step_height: float,
) -> None:
    """Render both sides' decoration onto ``meshes`` (deterministic; no RNG)."""
    kw = dict(
        size_x=size_x, x_offset=x_offset, y_center=y_center, stair_width=stair_width,
        z_max=z_max, z_min=z_min, step_height=step_height,
    )
    for side, is_left in ((decor.left, True), (decor.right, False)):
        if side.kind == "wall":
            _add_wall_one_side(meshes, side, left=is_left, **kw)
        elif side.kind == "posts":
            _add_posts_one_side(meshes, side, left=is_left, **kw)


def _build_stair_steps(
    cfg,
    *,
    size_x: float,
    size_y: float,
    going_up: bool,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    step_dim_override: tuple[float, float] | None = None,
) -> tuple[list[trimesh.Trimesh], np.ndarray, float, float, int, float, float | None]:
    """Build the per-step mesh + tread-center geometry for one staircase span.

    Shared by the legacy ``_build_stairs`` (whole sub-terrain) and by
    :class:`StairBlock` (per-block span inside a composite sub-terrain). The
    function works in **sub-terrain-local corner coordinates** and shifts all
    geometry by ``(x_offset, y_offset)`` so callers can place the span at an
    arbitrary x-position within the parent sub-terrain.

    Args:
        cfg: Object with the stair geometry parameters (``step_dim_options``,
            ``stair_width_range``, ``start_z_zero``, ``float_prob``,
            ``float_thick_range``, ``wall_prob``, ``pole_prob``,
            ``wall_thickness``, ``wall_height``, ``pole_thickness_range``,
            ``pole_spacing_range``, ``pole_height``,
            ``allow_partial_last_step``).
        size_x: x extent of the stair span (m).
        size_y: y extent of the stair span (m).
        going_up: ``True`` for stair 0 lowest (ascending in +x); ``False`` for
            stair 0 highest.
        x_offset: x coordinate of the span's bottom-left corner.
        y_offset: y coordinate of the span's bottom-left corner.

    Returns:
        ``(meshes, stair_top_centers, step_height, step_depth, num_steps,
        stair_width, partial_step_length)``. ``stair_top_centers`` is shape
        ``(num_steps, 3)`` with coordinates already shifted by
        ``(x_offset, y_offset, 0)``. ``partial_step_length`` is ``None`` when
        the trailing tread has the same depth as every other tread, or a
        positive float ``< step_depth`` when ``allow_partial_last_step`` is
        set on the cfg and the span doesn't tile exactly.
    """
    # 1. Sample step (height, depth) and stair width.  Callers that have
    # already pre-sampled the (h, d) pair (e.g. the composite ``StairBlock``
    # builder, which needs to validate the block's ``size_x`` against the
    # chosen step_depth before delegating here) can pass it via
    # ``step_dim_override`` to skip the resample.
    if step_dim_override is not None:
        step_height, step_depth_nominal = step_dim_override
    else:
        options = list(cfg.step_dim_options)
        if not options:
            raise ValueError("step_dim_options must contain at least one (height, depth) tuple")
        step_height, step_depth_nominal = options[np.random.randint(len(options))]
    step_height = float(step_height)
    step_depth_nominal = float(step_depth_nominal)
    stair_width = float(np.random.uniform(*cfg.stair_width_range))

    # 2-3. Step depth/count, vertical anchoring, and per-tread centers come from the shared,
    # isaaclab-free ``stair_tread_layout`` -- the SINGLE SOURCE OF TRUTH also used by the offline
    # clip generator, so a generated descent lands on precisely these treads.  This block only
    # wraps them with the box meshes + decoration below.
    layout = stair_tread_layout(
        size_x=size_x, size_y=size_y, step_height=step_height,
        step_depth_nominal=step_depth_nominal, going_up=going_up,
        x_offset=x_offset, y_offset=y_offset,
        start_z_zero=bool(getattr(cfg, "start_z_zero", False)),
        allow_partial_last_step=bool(getattr(cfg, "allow_partial_last_step", False)),
    )
    step_depth = layout.step_depth
    num_steps = layout.num_steps
    partial_step_length: float | None = layout.partial_step_length
    z_lowest_top, z_highest_top = layout.z_lowest_top, layout.z_highest_top
    y_center = y_offset + size_y / 2.0                      # span centre-line (for side decor)

    # 4. Resolve decoration (floating / per-side walls-or-posts / nosing).  When
    # ``cfg.decor`` is set this is deterministic (offline clip path); otherwise it
    # is sampled once from the cfg probabilities (online / legacy path).
    decor = _resolve_decor(cfg)
    is_floating = decor.float_thickness > 0.0
    floating_thickness = float(decor.float_thickness)

    meshes: list[trimesh.Trimesh] = []
    stair_top_centers = layout.tread_top_centers.copy()

    for i in range(num_steps):
        # Order along +x: stair 0 is at the span's xmin. For up stairs stair 0
        # is lowest; for down stairs stair 0 is highest.  Centers/depths precomputed above.
        cx, cy, top_z = (float(v) for v in stair_top_centers[i])
        this_depth = float(layout.step_depths[i])

        if is_floating:
            dims = (this_depth, stair_width, floating_thickness)
            cz = top_z - floating_thickness / 2.0
        else:
            # Solid step: extend down to one step below the lowest stair top so
            # neighbouring boxes always overlap with no visible gap underneath.
            box_bottom = z_lowest_top - step_height
            box_h = top_z - box_bottom
            dims = (this_depth, stair_width, box_h)
            cz = (top_z + box_bottom) / 2.0

        meshes.append(
            trimesh.creation.box(dims, trimesh.transformations.translation_matrix((cx, cy, cz)))
        )

        # Rounded leading nosing on this tread (the exposed top-front edge over the
        # riser): -x edge for an up-stair, +x edge for a down-stair.
        if decor.nosing_radius > 0.0:
            front_x = (cx - this_depth / 2.0) if going_up else (cx + this_depth / 2.0)
            nose = _rounded_nosing(
                front_x, cy, top_z, stair_width, decor.nosing_radius,
                decor.nosing_segments, step_height, this_depth,
            )
            if nose is not None:
                meshes.append(nose)

    # 5. Per-side decoration (independent walls / posts / none on each side).
    _apply_decor(
        meshes,
        decor,
        size_x=size_x,
        x_offset=x_offset,
        y_center=y_center,
        stair_width=stair_width,
        z_max=z_highest_top,
        z_min=z_lowest_top,
        step_height=step_height,
    )

    return (
        meshes,
        stair_top_centers,
        step_height,
        step_depth,
        num_steps,
        stair_width,
        partial_step_length,
    )


def _build_stairs(
    cfg,
    going_up: bool,
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Shared body for the up and down legacy stair generators.

    The function works in **local corner coordinates** (x in [0, size_x], y in
    [0, size_y]); :class:`MetaTerrainGenerator` re-centers the mesh and origin
    afterwards. The projection callable in metadata also operates in this
    frame.
    """
    size_x = float(cfg.size[0])
    size_y = float(cfg.size[1])

    meshes, stair_top_centers, step_height, step_depth, num_steps, stair_width, _ = (
        _build_stair_steps(cfg, size_x=size_x, size_y=size_y, going_up=going_up)
    )

    # Env origin — middle of a stair top, ~origin_distance_from_back from x=0.
    origin_idx = int(cfg.origin_distance_from_back // step_depth)
    origin_idx = max(0, min(num_steps - 1, origin_idx))
    origin = np.array(stair_top_centers[origin_idx], dtype=np.float64)

    # Metadata.
    centers_t = torch.from_numpy(stair_top_centers).clone()
    meta_data = {
        "needs_projection": False,
        "needs_directional_cmd": False,
        "is_border": False,
        "stair_dimension": (step_height, step_depth, stair_width),
        "num_steps": num_steps,
        "direction": "up" if going_up else "down",
        "stair_top_centers": centers_t,
        "project": _make_project_fn(centers_t, step_depth, num_steps),
        # Sampling probability per skill — must sum to 1.0.
        "skill_probs": {"stair_up": 1.0} if going_up else {"stair_down": 1.0},
        "velocity_guidance": cfg.velocity_guidance,
    }
    return meshes, origin, meta_data


def x_stairs_up_terrain(
    difficulty: float, cfg: "MeshXStairsUpTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Generate a staircase ascending in +x, vertically centered at z=0.

    Args:
        difficulty: Unused; kept for the standard sub-terrain function signature.
        cfg: Up-stairs configuration.

    Returns:
        Tuple of (meshes, origin, metadata) in local corner coordinates.
    """
    del difficulty
    return _build_stairs(cfg, going_up=True)


def x_stairs_down_terrain(
    difficulty: float, cfg: "MeshXStairsDownTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Generate a staircase descending in +x, vertically centered at z=0.

    Args:
        difficulty: Unused; kept for the standard sub-terrain function signature.
        cfg: Down-stairs configuration.

    Returns:
        Tuple of (meshes, origin, metadata) in local corner coordinates.
    """
    del difficulty
    return _build_stairs(cfg, going_up=False)
