# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain tile SAMPLING policy -- isaaclab-free single source of truth.

The RL terrain orchestrator (``robot_rl`` ``terrain_clip_orchestrator``) BUILDS geometry from
these specs (needs IsaacLab); the offline clip generator SAMPLES the same tile population to build
clips on. Both import ``TerrainSpec`` / ``sample_terrain_specs`` / ``split_counts`` from here so the
sampling policy is shared and the generator stays isaaclab-free. Pure ``numpy`` (+ ``resolve_stair_decor``).

``sample_terrain_specs`` samples two tile families:
    * ``ttype="terrain"`` -- a single long staircase, NO platforms (pure stairs, for long looped clips).
    * ``ttype="transition"`` -- flat lead-in -> staircase(s) -> top platform (learning on/off transitions).
      ``num_transitions_range`` may be ``(1, 1)`` for a single staircase per tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from motion_gen.motion_matching.terrain_variants import resolve_stair_decor


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
    stair_decor: list[dict] = field(default_factory=list)  # resolved decor per stair band (aligned w/ stair_steps)


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
    decor_params: dict | None = None,
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

    # Stair decoration (floating / walls / posts / nosing).  Only draw when a knob is
    # actually on, so an all-zero spec consumes NO rng and reproduces prior libraries
    # byte-for-byte.  When enabled, resolve one decor dict per stair band.
    _dp = decor_params or {}
    _decor_on = any(float(_dp.get(k, 0.0)) > 0.0 for k in
                    ("float_frac", "side_wall_frac", "side_posts_frac", "nosing_frac"))

    def _decor(n_bands: int) -> list[dict]:
        """Resolve ``n_bands`` stair-decor dicts (empty when decor is disabled)."""
        return [resolve_stair_decor(rng, **_dp) for _ in range(n_bands)] if _decor_on else []

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
            nav_width=w, spawn_y_offset=0.0, stair_decor=_decor(1),
        ))
        tid += 1

    min_lead, max_lead = lead_in_range
    min_inter, max_inter = inter_flat_range
    min_end = end_platform
    if max_lead < min_lead:
        raise ValueError(f"lead_in_range max {max_lead} < min {min_lead}")
    if max_inter < min_inter:
        raise ValueError(f"inter_flat_range max {max_inter} < min {min_inter}")
    nt_lo = max(1, num_transitions_range[0])                 # >= 1 staircase (pass (1,1) for a single flight)
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
        spawn_y_offset = float(rng.uniform(-off_max, off_max))
        specs.append(TerrainSpec(
            terrain_id=tid, ttype="transition", step_height=float(h), step_depth=float(d),
            width=tread_w, lead_in=lead_in, stair_steps=steps, flat_after=flat_after,
            nav_width=nav_width, spawn_y_offset=spawn_y_offset, stair_decor=_decor(len(steps)),
        ))
        tid += 1
    return specs


def sample_stair_tiles(
    rng: np.random.Generator,
    *,
    hd_grid: list[tuple[float, float]],
    n_pure: int,
    n_transition: int,
    pure_length: float,
    transition_length: float,
    width_range: tuple[float, float],
    num_steps_range: tuple[int, int],
    lead_in_range: tuple[float, float],
    end_platform: float,
    inter_flat_range: tuple[float, float],
    nav_width_range: tuple[float, float],
    flat_width_margin: float,
    single_step_frac: float = 0.0,
    decor_params: dict | None = None,
) -> list[TerrainSpec]:
    """Sample the offline stair TILE population -- phase A of the tiles-then-clips generator.

    Two families, sized independently (the pure tiles must be much longer to hold a long looped
    descent, so they get their own ``pure_length`` rather than sharing ``transition_length``):

    * ``n_pure`` ``"terrain"`` tiles -- a single long, platform-less staircase spanning ``pure_length``
      (``~pure_seconds x stair_vx``); a looped descent clip runs its whole length without hitting an end.
    * ``n_transition`` ``"transition"`` tiles -- flat lead-in -> a SINGLE staircase -> top platform, for
      learning the on/off transitions.  ``single_step_frac`` of them are single-riser curbs.

    Both draw dims from ``hd_grid``, tread width from ``width_range``, step count from
    ``num_steps_range`` (transitions), and decoration from ``decor_params``.  The merged list is
    re-numbered so ``terrain_id`` is contiguous ``0..N-1`` (each tile is reused by many clips
    downstream, so one id == one physical tile).
    """
    common = dict(
        hd_grid=hd_grid, width_range=width_range, num_steps_range=num_steps_range,
        lead_in_range=lead_in_range, end_platform=end_platform, num_transitions_range=(1, 1),
        inter_flat_range=inter_flat_range, nav_width_range=nav_width_range,
        flat_width_margin=flat_width_margin, decor_params=decor_params,
    )
    pure = sample_terrain_specs(rng, n_terrain=n_pure, n_transition=0,
                                transition_length=pure_length, **common)
    trans = sample_terrain_specs(rng, n_terrain=0, n_transition=n_transition,
                                 transition_length=transition_length,
                                 single_step_frac=single_step_frac, **common)
    tiles = pure + trans
    for i, t in enumerate(tiles):
        t.terrain_id = i                                     # contiguous ids across the merged set
    return tiles
