# Copyright 2026 Zachary Olkin. All rights reserved.

"""Orchestrate terrain-aware clip generation across a fixed set of terrains.

The RL env can only load a bounded number of terrains, so a whole clip library
is generated over ``num_terrains`` terrains: the requested fractions split them
into ``flat`` / ``terrain`` (terrain-only) / ``transition`` (flat + one or more
staircases), and each non-flat terrain hosts ``clips_per_terrain`` clips at
sampled velocities.  Every terrain has *uniform* stair dimensions (sampled from
the library's discrete stair grid) so its stair-dim trajectory filter and
projector are constant for the whole batch.

Geometry is a **1-D corridor** of full-width bands chained along +x, reusing the
terrain "block" system (:func:`composite_terrain` + :class:`StairBlockCfg` /
:class:`FlatBlockCfg`):

* ``terrain``    : ``[flat base][stairs][flat top]`` (one staircase; spawn on it).
* ``transition`` : ``[flat lead-in]([stairs][flat])xN`` -- one or more flat<->stair
  transitions.  A transition clip spawns at a start corner and its lateral / yaw
  command is PD-steered toward a single fixed goal at the far centerline end
  (see :meth:`OfflineClipGenerator._apply_goal_steer`), giving a turn while
  keeping the robot on the terrain.  Flat bands span the full (wider) corridor
  width; stairs are narrower and centered.

The per-terrain geometry (all bands + the nav goal) is recorded in a manifest
embedded in the clip ``.h5`` so viser and the plots can rebuild it.

This module holds no CLI; :mod:`scripts.rsl_rl.generate_terrain_clips` builds the
manager / samplers / writer and drives :func:`orchestrate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.composite import (
    CompositeSubTerrainCfg,
    composite_terrain,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.flat import FlatBlockCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.blocks.stair import StairBlockCfg

from motion_gen.clips.offline_clip_generator import OfflineClipGenerator
from .offline_stair_projector import make_offline_stair_projector, make_offline_multi_stair_projector

# Small x-clearance placing the nav goal before the far edge (m).
_GOAL_MARGIN = 0.3


@dataclass
class TerrainSpec:
    """One uniform terrain (corridor of bands) to build clips on."""

    terrain_id: int
    ttype: str                     # "terrain" | "transition"
    step_height: float
    step_depth: float
    width: float                   # stair tread width in y (< nav_width)
    lead_in: float                 # first flat band length (x): base / walk lead-in
    stair_steps: list[int]         # num steps per stair band (len == num_transitions)
    flat_after: list[float]        # flat band length after each stair band (last = top platform)
    nav_width: float               # corridor width in y (transition nav); == width for terrain-only
    spawn_y_offset: float          # signed lateral spawn offset from centerline (transition turn)


@dataclass
class BuiltTerrain:
    """Geometry + runtime hooks for a built terrain."""

    spec: TerrainSpec
    projector: object                         # project(points[N,3]) -> [N,3] (multi-band)
    band_projectors: list                     # per-band single-span project(points[N,3]) -> [N,3]
    dim_lookup: object                        # fn(xy) -> (dims[N,3], valid[N])
    spawn_anchor_pos: torch.Tensor            # [3]
    spawn_on_stairs: bool
    spawn_ground_z: float
    band_exit_z: list[float]                  # per stair-band top-of-stairs z (ascending)
    steer_to_goal: bool                       # transition nav
    nav_goal_xy: list[float] | None           # [2] fixed goal (local-corner), transition only
    manifest: dict = field(default_factory=dict)


def split_counts(num_terrains: int, fracs: dict[str, float]) -> dict[str, int]:
    """Split ``num_terrains`` into per-type counts from (normalized) fractions.

    Rounds each type and assigns the rounding remainder to the largest type so
    the counts sum exactly to ``num_terrains``.

    Args:
        num_terrains: Total number of terrains to allocate.
        fracs: ``{"flat": f, "terrain": f, "transition": f}`` (need not sum to 1).

    Returns:
        ``{"flat": n, "terrain": n, "transition": n}`` summing to ``num_terrains``.
    """
    total = sum(fracs.values())
    if total <= 0:
        raise ValueError(f"terrain fractions must be positive; got {fracs}")
    counts = {k: int(round(num_terrains * v / total)) for k, v in fracs.items()}
    drift = num_terrains - sum(counts.values())
    if drift != 0:
        biggest = max(fracs, key=lambda k: fracs[k])
        counts[biggest] += drift
    return counts


def sample_terrain_specs(
    rng: np.random.Generator,
    n_terrain: int,
    n_transition: int,
    hd_grid: list[tuple[float, float]],
    width_range: tuple[float, float],
    num_steps_range: tuple[int, int],
    lead_in_range: tuple[float, float],
    end_platform: float,
    num_transitions_range: tuple[int, int],
    inter_flat_range: tuple[float, float],
    nav_width_range: tuple[float, float],
    flat_width_margin: float,
    transition_length: float,
    single_step_frac: float = 0.0,
) -> list[TerrainSpec]:
    """Sample per-terrain band chains for the terrain + transition sets.

    No stair start-platform is used (the first riser sits at each stair block's
    left edge, so an approaching foot snapped to ``span_xmin - depth/2`` is one
    step short of tread 0); a flat lead-in / inter-stair flat provides the
    ground before each staircase.  ``terrain`` uses a short base + single
    staircase and NOTHING else -- no flat base or top platform, just the stairs;
    the robot spawns on the ambient ground right before the first riser and
    climbs).  ``transition`` uses >= 2 staircases separated by
    flat bands and is built to a FIXED total length ``transition_length``: the
    staircases are sampled freely (varied step count) and the TOP platform absorbs
    the remaining length, so every transition terrain is the same size_x while the
    stairs still vary.  The lead-in (start platform) is sampled in ``lead_in_range``
    ``[min, max]`` and the inter-stair flats in ``inter_flat_range`` ``[min, max]``
    (so a flat between two staircases is always in that range -- the slack does NOT
    pile onto them); the TOP platform (last ``flat_after``) takes whatever is left to
    hit ``transition_length`` (floored at ``end_platform``).  If the sampled stairs +
    flats overrun the fixed length, step counts are trimmed first (kept >=
    ``num_steps_range[0]``), then the lead-in / inter-flat slack above their floors is
    shaved -- inter-flats stay within ``inter_flat_range``.

    ``single_step_frac`` of the transitions are SINGLE-STEP curbs: one staircase of
    one riser with a flat lead-in and a long top platform (step up a curb, then walk),
    for learning small step-ups independent of full staircases.

    ``width_range`` is the stair tread width ``(lo, hi)``: a fixed-width library
    has ``lo == hi`` (every terrain uses that width), a range library (a stair
    trajectory declaring ``terrain_width`` as a ``{min, max}`` range) is sampled
    per terrain so terrains vary in stair width.  For a ``transition`` the tread
    width is additionally capped at that terrain's corridor width ``nav_width``
    so the stairs never grow wider than the flat corridor leading to them.
    """
    specs: list[TerrainSpec] = []
    tid = 0
    w_lo, w_hi = float(width_range[0]), float(width_range[1])

    def _sample_width(cap: float | None = None) -> float:
        """Uniform tread width in the range (fixed when lo == hi), capped."""
        hi = w_hi if cap is None else min(w_hi, cap)
        hi = max(hi, w_lo)
        return float(rng.uniform(w_lo, hi)) if hi > w_lo else w_lo

    for _ in range(n_terrain):
        h, d = hd_grid[int(rng.integers(len(hd_grid)))]
        w = _sample_width()
        # One big staircase spanning ~transition_length (so terrain-only terrains
        # are the same length as transitions and the robot climbs for the whole
        # clip); just the staircase, no flat base / top.
        ns = max(2, int(round(transition_length / d)))
        specs.append(TerrainSpec(
            terrain_id=tid, ttype="terrain", step_height=float(h), step_depth=float(d),
            width=w, lead_in=0.0, stair_steps=[ns], flat_after=[0.0],
            nav_width=w, spawn_y_offset=0.0,
        ))
        tid += 1

    min_lead, max_lead = lead_in_range
    min_inter, max_inter = inter_flat_range
    min_end = end_platform
    if max_lead < min_lead:
        raise ValueError(f"lead_in_range max {max_lead} < min {min_lead}")
    if max_inter < min_inter:
        raise ValueError(f"inter_flat_range max {max_inter} < min {min_inter}")
    nt_lo = max(2, num_transitions_range[0])                 # >= 2 staircases per transition
    nt_hi = max(nt_lo, num_transitions_range[1])
    for _ in range(n_transition):
        h, d = hd_grid[int(rng.integers(len(hd_grid)))]
        if rng.random() < single_step_frac:
            n, steps = 1, [1]                                # single-step curb: one riser + platform
        else:
            n = int(rng.integers(nt_lo, nt_hi + 1))
            steps = [int(rng.integers(num_steps_range[0], num_steps_range[1] + 1)) for _ in range(n)]
        # Lead-in + inter-stair flats are sampled IN their ranges; the TOP platform absorbs the slack
        # to hit the fixed transition_length (so inter-stair flats stay bounded, not slack-inflated).
        lead_in = float(rng.uniform(min_lead, max_lead))
        inter = rng.uniform(min_inter, max_inter, size=max(0, n - 1))

        def _end():                                          # top-platform length that fills to L
            return transition_length - lead_in - float(sum(steps)) * d - float(inter.sum())

        # If the draw overruns L, trim steps first (kept >= min), then shave lead-in / inter slack
        # above their floors -- inter-flats stay within [min_inter, max_inter].
        while _end() < min_end and any(s > num_steps_range[0] for s in steps):
            above = [i for i, s in enumerate(steps) if s > num_steps_range[0]]
            steps[above[int(rng.integers(len(above)))]] -= 1
        if _end() < min_end:
            over = min_end - _end()
            slack = (lead_in - min_lead) + float((inter - min_inter).sum())
            if slack > 1e-9:
                fr = min(1.0, over / slack)
                lead_in -= fr * (lead_in - min_lead)
                inter = inter - fr * (inter - min_inter)
        if _end() < min_end - 1e-6:
            raise ValueError(
                f"--transition-length {transition_length:.2f}m too short for {n} staircase(s) of "
                f">= {num_steps_range[0]} steps at depth {d:.3f}m + flat floors."
            )
        lead_in = float(lead_in)
        flat_after = [float(x) for x in inter] + [float(_end())]

        nav_width = float(rng.uniform(*nav_width_range))
        off_max = max(0.0, nav_width / 2.0 - flat_width_margin)
        # Tread width sampled in the range, capped at this corridor so the
        # stairs stay within the flat corridor leading up to them.
        tread_w = _sample_width(cap=nav_width)
        specs.append(TerrainSpec(
            terrain_id=tid, ttype="transition", step_height=float(h), step_depth=float(d),
            width=tread_w, lead_in=lead_in, stair_steps=steps, flat_after=flat_after,
            nav_width=nav_width, spawn_y_offset=float(rng.uniform(-off_max, off_max)),
        ))
        tid += 1
    return specs


def build_terrain(spec: TerrainSpec, geom_seed: int, device, size_y_floor: float = 0.0) -> BuiltTerrain:
    """Build one terrain's geometry, projector, dim-lookup, spawn/goal, manifest.

    Args:
        spec: The terrain to build.
        geom_seed: Seed for the block sampler (``np.random``) so any residual
            block-internal randomness is reproducible (dims are fixed by ``spec``).
        device: Torch device for the projector constants / spawn anchor.
        size_y_floor: Lower bound on the corridor width (``--size-y``).

    Returns:
        A :class:`BuiltTerrain`.
    """
    np.random.seed(geom_seed)
    size_y = float(max(spec.nav_width, spec.width, size_y_floor))

    # Band chain: optional [flat lead-in] then [stairs][flat] per transition.
    # Zero-length flat bands are skipped (terrain-only uses no base / no top
    # platform -- it is JUST the staircase).
    blocks: list = []
    if spec.lead_in > 0.0:
        blocks.append(FlatBlockCfg(size_x=spec.lead_in))
    for ns, fa in zip(spec.stair_steps, spec.flat_after):
        blocks.append(StairBlockCfg(
            size_x=ns * spec.step_depth, direction="up",
            step_dim_options=[(spec.step_height, spec.step_depth)],
            stair_width_range=(spec.width, spec.width),
            start_platform_length=0.0, end_platform_length=0.0,
        ))
        if fa > 0.0:
            blocks.append(FlatBlockCfg(size_x=fa))
    size_x = float(sum(b.size_x for b in blocks))
    cfg = CompositeSubTerrainCfg(size=(size_x, size_y), blocks=blocks, origin_block_index=0, start_z=0.0)
    _meshes, _origin, meta = composite_terrain(0.0, cfg)

    # Per stair band: span (for the projector) + exit z (for the climb top).
    spans: list[dict] = []
    band_exit_z: list[float] = []
    for b in meta["blocks"]:
        if not b["extras"].get("is_stair"):
            continue
        centers = b["extras"]["stair_top_centers"].to(device=device, dtype=torch.float32)
        spans.append({
            "stair_top_centers": centers,
            "step_depth": float(b["extras"]["stair_dimension"][1]),
            "entry_z": float(b["entry_z"]),
            "span_xmin": float(b["aabb"][0]),
            "span_xmax": float(b["aabb"][1]),
        })
        band_exit_z.append(float(b["exit_z"]))
    projector = make_offline_multi_stair_projector(spans)
    # Per-band single-span projectors: the generator projects a clip's foot with
    # the projector for the band it is climbing / approaching (``band_idx``),
    # NOT the multi-span "next stair ahead" -- the latter routes a foot that has
    # slightly overshot a staircase's top (x past span_xmax) onto the NEXT
    # staircase, teleporting the clip forward over the flat.
    band_projectors = [
        make_offline_stair_projector(
            s["stair_top_centers"], s["step_depth"], s["entry_z"], s["span_xmin"], s["span_xmax"],
        )
        for s in spans
    ]

    # Constant terrain-dim lookup (width unconstrained via NaN, matches online).
    dims_const = torch.tensor([float("nan"), spec.step_height, spec.step_depth], device=device)

    def dim_lookup(xy: torch.Tensor):
        n = xy.shape[0]
        return dims_const.unsqueeze(0).expand(n, 3), torch.ones(n, dtype=torch.bool, device=xy.device)

    cy = size_y / 2.0
    if spec.ttype == "transition":
        # PD goal-steering is disabled for now (kept dormant in the generator):
        # transition clips walk STRAIGHT up the centerline (vy = wz = 0) so each
        # staircase is crossed head-on.  Spawn centered on a flat lead-in.
        spawn_x = min(0.5, max(0.1, spec.lead_in * 0.25))
        spawn_pos = torch.tensor([spawn_x, cy, 0.0], device=device)
        spawn_on_stairs, spawn_ground_z, steer = False, 0.0, False
        nav_goal_xy = None
    else:  # terrain-only: JUST the staircase (no base) -> spawn ON the first
        # step (stance foot on tread 0), so the robot starts already climbing.
        tread0 = spans[0]["stair_top_centers"][0]  # [x, y, z] of the lowest tread top
        spawn_pos = torch.tensor([float(tread0[0]), cy, float(tread0[2])], device=device)
        spawn_on_stairs, spawn_ground_z, steer = True, spans[0]["entry_z"], False
        nav_goal_xy = None

    # Manifest: all bands (for viser meshes) + nav goal; legacy band-0 stair
    # keys kept for backward-compatible readers (plots / old viz).
    band_meta = []
    for b in meta["blocks"]:
        entry = {"block_type": b["block_type"], "aabb": list(b["aabb"]),
                 "entry_z": float(b["entry_z"]), "exit_z": float(b["exit_z"])}
        if b["extras"].get("is_stair"):
            entry["stair_top_centers"] = b["extras"]["stair_top_centers"].cpu().tolist()
            entry["num_steps"] = int(b["extras"]["num_steps"])
        band_meta.append(entry)
    sb0 = spans[0]
    manifest = {
        "terrain_id": spec.terrain_id,
        "type": spec.ttype,
        "dims": {"h": spec.step_height, "d": spec.step_depth, "w": spec.width},
        "num_transitions": len(spec.stair_steps),
        "stair_width": spec.width,
        "size_x": size_x,
        "size_y": size_y,
        "spawn_anchor": spawn_pos.cpu().tolist(),
        "nav_goal": nav_goal_xy,
        "bands": band_meta,
        # Legacy single-stair keys (band 0) for backward-compatible readers.
        "num_steps": spec.stair_steps[0],
        "entry_z": sb0["entry_z"],
        "exit_z": band_exit_z[-1],
        "span_xmin": sb0["span_xmin"],
        "span_xmax": sb0["span_xmax"],
        "stair_top_centers": sb0["stair_top_centers"].cpu().tolist(),
    }
    return BuiltTerrain(
        spec=spec, projector=projector, band_projectors=band_projectors, dim_lookup=dim_lookup,
        spawn_anchor_pos=spawn_pos, spawn_on_stairs=spawn_on_stairs,
        spawn_ground_z=spawn_ground_z, band_exit_z=band_exit_z,
        steer_to_goal=steer, nav_goal_xy=nav_goal_xy, manifest=manifest,
    )


def _terrain_xy_footprint(manifest: dict) -> tuple[list[tuple[float, float, float, float]], float]:
    """Per-band ground footprint rectangles ``(x0, x1, y_lo, y_hi)`` + ``size_x``.

    Flat bands span the full corridor width ``[0, size_y]``; stair bands span only
    the (narrower, centered) tread width ``w`` about their tread-center y.  A
    reference foot whose xy leaves its band's rectangle is off the walkable
    surface (off the tread edge, or past the corridor side).
    """
    size_y = float(manifest["size_y"])
    w = float(manifest["stair_width"])
    rects: list[tuple[float, float, float, float]] = []
    for b in manifest["bands"]:
        x0, x1 = float(b["aabb"][0]), float(b["aabb"][1])
        if b["block_type"] == "StairBlock":
            cy = float(np.mean(np.array(b["stair_top_centers"], dtype=float)[:, 1]))
            rects.append((x0, x1, cy - w / 2.0, cy + w / 2.0))
        else:
            rects.append((x0, x1, 0.0, size_y))
    return rects, float(manifest["size_x"])


def report_off_terrain(
    batch: dict,
    manifest: dict,
    *,
    tol: float = 1e-3,
    feet: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-clip check that no reference foot leaves the terrain's lateral footprint.

    For each foot at each step, the band under its forward ``x`` sets the allowed
    lateral ``y`` extent (tread width on stairs, full corridor on flats); the
    excursion is how far ``y`` lies beyond that extent (0 if inside).  Longitudinal
    (off the front/back) is intentionally not flagged -- a foot naturally overhangs
    the top tread -- the concern is lateral drift off the stairs / corridor.

    Args:
        batch: A :meth:`OfflineClipGenerator.generate_batch` output (uses
            ``y_des`` and ``meta["pos_output_names"]``).
        manifest: The built terrain manifest (``bands``, ``size_y``, ``stair_width``).
        tol: Lateral slack (m) before a foot counts as off the edge.
        feet: Frame names checked (both ankles by default).

    Returns:
        ``(off_mask[n] bool, worst_exc[n])`` -- per clip, whether any foot went off
        the footprint (beyond ``tol``) and the worst lateral excursion (m).
    """
    y = batch["y_des"]
    names = batch["meta"]["pos_output_names"]
    if not manifest.get("bands"):
        n = y.shape[0]
        return torch.zeros(n, dtype=torch.bool, device=y.device), torch.zeros(n, device=y.device)
    rects, _size_x = _terrain_xy_footprint(manifest)
    dev = y.device
    edges = torch.tensor([rects[0][0]] + [r[1] for r in rects], device=dev)  # contiguous band boundaries
    y_lo = torch.tensor([r[2] for r in rects], device=dev)
    y_hi = torch.tensor([r[3] for r in rects], device=dev)
    n = y.shape[0]
    off = torch.zeros(n, dtype=torch.bool, device=dev)
    worst = torch.zeros(n, device=dev)
    for f in feet:
        if f"{f}:pos_x" not in names:
            continue
        fx = y[:, :, names.index(f"{f}:pos_x")].contiguous()          # [n, T]
        fy = y[:, :, names.index(f"{f}:pos_y")]
        band = torch.bucketize(fx, edges[1:-1]).clamp_(0, len(rects) - 1)  # [n, T] band index
        exc = torch.clamp(y_lo[band] - fy, min=0.0) + torch.clamp(fy - y_hi[band], min=0.0)
        off |= (exc > tol).any(dim=1)
        worst = torch.maximum(worst, exc.max(dim=1).values)
    return off, worst


def orchestrate(
    *,
    generator: OfflineClipGenerator,
    samplers: dict[str, object],
    writer,
    counts: dict[str, int],
    specs: list[TerrainSpec],
    clips_per_terrain: int,
    size_y_floor: float,
    seed: int,
    device,
    progress: bool = True,
    approach_skills: tuple[str, ...] | None = None,
) -> list[dict]:
    """Generate and write the whole terrain-aware clip library.

    Flat clips (``counts["flat"] * clips_per_terrain``) use the flat rollout;
    each terrain in ``specs`` gets ``clips_per_terrain`` clips via its own bound
    terrain context.  Returns the terrain manifest (list of dicts).

    Args:
        generator: The (single) clip generator; its ``sampler`` is swapped per
            batch and its terrain context bound per terrain.
        samplers: ``{"flat", "terrain", "transition"}`` velocity samplers sharing
            the generator's bucket tables (same buckets).
        writer: An open :class:`ClipLibraryWriter` sized to the total clip count.
        counts: Per-type terrain counts (from :func:`split_counts`).
        specs: The terrain + transition specs (from :func:`sample_terrain_specs`).
        clips_per_terrain: Clips per terrain (also per flat terrain slot).
        size_y_floor: Lower bound on the corridor width.
        seed: Base seed; per-terrain geometry seeds derive from it.
        device: Torch device.
        progress: Print per-terrain progress.
        approach_skills: Skill names the approach slow-down may use on every
            terrain/transition (forward-walking skill(s)); ``None`` leaves the
            generator's fallback (flat-band mask).

    Returns:
        The terrain manifest list (also written to the ``.h5`` by the caller).
    """
    manifest: list[dict] = []
    written = 0

    def _emit(n_clips: int, tag: str, on_batch=None) -> None:
        nonlocal written
        remaining = n_clips
        while remaining > 0:
            b = generator.generate_batch(remaining)
            if on_batch is not None:
                on_batch(b)
            writer.write_batch(written, b)
            written += b["y_des"].shape[0]
            remaining -= b["y_des"].shape[0]
        if progress:
            print(f"  [{tag}] {written} clips written", flush=True)

    # --- Flat clips (no terrain) ---
    n_flat_clips = counts.get("flat", 0) * clips_per_terrain
    if n_flat_clips > 0:
        generator.clear_terrain_context()
        generator.sampler = samplers["flat"]
        _emit(n_flat_clips, "flat")

    # --- Terrain + transition clips (one bound terrain per spec) ---
    for spec in specs:
        built = build_terrain(spec, geom_seed=seed + 1000 + spec.terrain_id, device=device, size_y_floor=size_y_floor)
        generator.set_terrain_context(
            projector=built.projector,
            band_projectors=built.band_projectors,
            terrain_dim_lookup=built.dim_lookup,
            spawn_anchor_pos=built.spawn_anchor_pos,
            spawn_anchor_quat=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
            spawn_on_stairs=built.spawn_on_stairs,
            spawn_ground_z=built.spawn_ground_z,
            band_exit_z=built.band_exit_z,
            terrain_id=spec.terrain_id,
            terrain_type=spec.ttype,
            steer_to_goal=built.steer_to_goal,
            nav_goal_xy=built.nav_goal_xy,
            # Non-steered terrain/transition clips walk straight up +x: lay every
            # stride world-aligned (global 0 yaw) so per-stride yaw residuals never
            # accumulate into a lateral drift off the corridor.
            straight_only=not built.steer_to_goal,
            # No running on the flat bands of a terrain (a slow command must not
            # mis-bucket onto a fast running gait); walk / stand only.
            flat_exclude_skills=("running",),
            # Approach slow-down restricted BY SKILL to the forward-walking
            # skill(s) so it never picks a backward / running / off-nominal stride.
            approach_skills=approach_skills,
        )
        generator.sampler = samplers[spec.ttype]
        # Track any reference foot that drifts off the terrain's lateral footprint
        # (off a stair tread / off the corridor) across this terrain's batches.
        off_stats = {"n_off": 0, "n_total": 0, "worst": 0.0}

        def _check_off_terrain(b, _m=built.manifest, _s=off_stats) -> None:
            off, worst = report_off_terrain(b, _m)
            _s["n_off"] += int(off.sum())
            _s["n_total"] += int(off.numel())
            _s["worst"] = max(_s["worst"], float(worst.max()) if worst.numel() else 0.0)

        _emit(clips_per_terrain, f"{spec.ttype}#{spec.terrain_id}", on_batch=_check_off_terrain)
        if off_stats["n_off"] > 0:
            bar = "!" * 78
            print(
                f"\n{bar}\n"
                f"!!  WARNING: {off_stats['n_off']}/{off_stats['n_total']} clip(s) on terrain "
                f"{spec.terrain_id} ({spec.ttype}) have a REFERENCE FOOT OFF THE EDGE OF THE\n"
                f"!!  TERRAIN -- worst lateral excursion {off_stats['worst']:.3f} m beyond the "
                f"tread / corridor edge.\n"
                f"!!  The stitched reference drifts laterally off the stairs/platform, so the robot\n"
                f"!!  would be told to step where there is no ground. Inspect with\n"
                f"!!  scripts/rsl_rl/plot_clip_anchor_drift.py (lateral drift), and check the corridor\n"
                f"!!  width (--nav-width-*) / crooked source strides ([yaw-diag]).\n"
                f"{bar}\n",
                flush=True,
            )
        manifest.append(built.manifest)

    generator.clear_terrain_context()
    return manifest
