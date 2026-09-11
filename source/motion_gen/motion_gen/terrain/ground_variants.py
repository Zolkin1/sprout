# Copyright 2026 Zachary Olkin. All rights reserved.

"""Vary the GROUND a recorded motion was walked on, without moving the motion.

``clutter.py`` adds visual-only boxes: the ground is untouched, so every recorded pose stays
valid by construction and the only thing that changes is what the cameras see. This changes
the ground itself, and exists because the generator has been measured conditioning on
features of the ground that are not the terrain:

* on one tile, adding a floor around an otherwise identical staircase flips the produced
  trajectory from stairs-down to jump-down;
* on another, it flips a slow controlled climb into the faster gait, which then falls.

Neither the treads nor the risers moved. What moved was the context, so the context is what
has to vary in training. Four edits, on stairs and boxes only -- never on flat, which has no
terrain to be confused about:

``floor``
    Present or absent, and when present **flush with the lowest walkable platform** rather
    than a riser below it. The shipped stair-transition tiles stand the robot on a plinth: the
    approach platform sits ~0.13 m above the surrounding ground, so a ledge runs the length of
    the corridor. That is a loud "this is a terrain tile" cue with nothing to do with the
    stairs, and it is absent from every box tile, which the robot walks onto straight off the
    floor. Flush removes it and makes the two families look alike where they ARE alike.
``width``
    Widen the flight. Narrow flights sit in a gap of missing ground, which identifies them
    without anything about the treads being read.
``truncate``
    End the landing platform at a sampled distance. Isaac's platforms run ~10 m; MuJoCo's
    playground stops at 2 m and descends, and a short platform reads as the top of a box.
``walls``
    Strip the library's own decoration, then re-add none / one side / both sides / across the
    platform end. Placed CLOSE -- along the flight, not at the tile edge. The RL-side clutter
    had to keep its distance because an episode can wander; a re-render cannot.

**Every edit is checked against the recorded feet.** Clutter needed a veto because it added
geometry the robot might walk into; this needs the mirror image, because it removes geometry
the robot may have been standing on. An edit is admissible only if every recorded foot still
has ground under it and none is left inside a newly raised surface. Widen never narrow,
truncate only beyond the furthest foot, raise a floor rather than lower a structure.

The result is a SIDECAR (schema 2), not a library: the motion is unchanged and copying
gigabytes of it to vary the ground would put the cost in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Vertical slack when asking "is this foot standing on that surface" (m). A foot resting on
#: a tread is not exactly at its height -- the ankle frame sits above the sole, the tracker
#: has error, and the recorded pose is a snapshot mid-step.
FOOT_TOL_M: float = 0.12

#: How far a wall stands from the geometry it flanks (m). Close, on purpose: this is fixed
#: recorded data, so nothing can wander into it.
WALL_GAP_M: float = 0.35

#: Wall thickness (m), matching the clutter pass so the two look alike to a camera.
WALL_T_M: float = 0.15

#: Treads are the short overlays; platforms are the long ones. Same threshold the MuJoCo tile
#: generator uses to tell a step from a slab.
STEP_MAX_RUN_M: float = 0.45


@dataclass
class VariantCfg:
    """Which edits a variant may make, and how far.

    Attributes:
        seed: Base seed; every tile draws from its own stream.
        floor_modes: Which floor treatments to draw from. ``"keep"`` leaves the tile as the
            library has it, ``"flush"`` puts ground at the lowest platform's top, ``"none"``
            removes it entirely (the Isaac look for a floating structure).
        widen_range: Multiplier on the flight's lateral extent. Never below 1.0 -- narrowing
            can pull ground out from under a recorded foot.
        truncate_frac_range: Where to end the landing platform, as a fraction of its own
            length past the last recorded foot. 1.0 leaves it alone.
        wall_modes: Wall arrangements to draw from.
    """

    seed: int = 0
    floor_modes: tuple[str, ...] = ("keep", "flush", "none")
    widen_range: tuple[float, float] = (1.0, 2.2)
    truncate_frac_range: tuple[float, float] = (0.15, 1.0)
    wall_modes: tuple[str, ...] = ("none", "one", "both", "end")
    strip_existing_decor: bool = True


def tile_rng(seed: int, terrain_id: int, salt: int = 0) -> np.random.Generator:
    """A generator private to one tile, so a variant is reproducible per tile."""
    return np.random.default_rng((int(seed) * 1_000_003 + int(terrain_id) + 2) ^ int(salt))


def walkable(overlays: list[dict]) -> list[int]:
    """Indices of the overlays the robot can stand on (everything not decoration)."""
    return [i for i, o in enumerate(overlays) if not o.get("decor")]


def is_step(o: dict) -> bool:
    """Whether an overlay is a tread rather than a platform."""
    return float(o["size_xyz"][0]) <= STEP_MAX_RUN_M


def top_z(o: dict) -> float:
    """Top surface height of an overlay (m)."""
    return float(o.get("base_z", 0.0)) + float(o["size_xyz"][2])


def aabb_xy(o: dict) -> tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)`` footprint of an overlay, in tile coordinates."""
    sx, sy, _ = (float(v) for v in o["size_xyz"])
    cx, cy = (float(v) for v in o["pos_xy"])
    return cx - sx / 2.0, cy - sy / 2.0, cx + sx / 2.0, cy + sy / 2.0


def feet_on(feet: np.ndarray, o: dict, tol: float = FOOT_TOL_M) -> np.ndarray:
    """Which feet are standing ON one overlay.

    Args:
        feet: ``(N, 3)`` tile-local foot positions.
        o: The overlay.
        tol: Vertical slack.

    Returns:
        ``(N,)`` bool.
    """
    x0, y0, x1, y1 = aabb_xy(o)
    inside = ((feet[:, 0] >= x0) & (feet[:, 0] <= x1)
              & (feet[:, 1] >= y0) & (feet[:, 1] <= y1))
    return inside & (np.abs(feet[:, 2] - top_z(o)) <= tol)


def flush_floor_z(overlays: list[dict], feet: np.ndarray, has_floor: bool,
                  pct: float = 1.0) -> float:
    """Where a flush floor goes: the top of the surface the robot actually started from.

    Two wrong answers this avoids, both found by measuring:

    * **the lowest walkable platform.** On a ``jump_on`` tile the box IS a platform by any
      geometric test, so this returns the box top and a floor there buries the approach the
      robot jumped from -- it stranded feet on 9 of 10 jump tiles and no stairs tile.
    * **the lowest recorded foot.** The ankle frame sits above the sole, so a foot standing
      on a 0.131 m platform records 0.165 m, and a floor at that height is a 34 mm plinth in
      the other direction.

    So the FEET choose which surface the motion began on and the GEOMETRY supplies its
    height. On a stairs transition that is the approach platform, and the plinth goes; on a
    jump tile it is the ground already, so the edit is correctly a no-op.

    Args:
        overlays: The tile's overlays.
        feet: ``(N, 3)`` tile-local recorded foot positions.
        has_floor: Whether the tile has ground of its own at ``z = 0``.
        pct: Percentile of foot height taken as "the low end of the motion".

    Returns:
        Height in metres; ``0.0`` when there is nothing to raise.
    """
    if feet.size == 0:
        return 0.0
    low = float(np.percentile(feet[:, 2], pct))
    tops = [top_z(overlays[i]) for i in walkable(overlays)]
    if has_floor:
        tops.append(0.0)
    # The highest surface AT OR BELOW that foot. The slack is small on purpose: a foot is
    # always above the thing it stands on, so the only allowance needed is numerical. Using
    # the standing tolerance (0.12 m) here instead reaches past the approach platform and
    # picks the SECOND TREAD, putting the floor a step up inside the flight.
    below = [t for t in tops if t <= low + 0.02]
    return max(below, default=0.0)

def _box(cx: float, cy: float, base_z: float, sx: float, sy: float, sz: float,
         *, decor: bool, cls: str | None = None) -> dict:
    """One overlay dict in the manifest's own schema."""
    o = {"kind": "box", "pos_xy": [float(cx), float(cy)],
         "size_xyz": [float(sx), float(sy), float(sz)],
         "base_z": float(base_z), "walkable_top": not decor, "yaw": 0.0}
    if decor:
        o["decor"] = True
        o["variant_class"] = cls or "wall"
    return o


def plan_variant(tile: dict, feet: np.ndarray, cfg: VariantCfg) -> dict:
    """Choose one tile's edits. Returns a schema-2 per-tile block.

    Nothing here is applied; the block is a statement of intent that
    :func:`check_feet` vets and the importer executes.

    Args:
        tile: The manifest entry.
        feet: ``(N, 3)`` tile-local recorded foot positions for this tile.
        cfg: What the variant is allowed to do.

    Returns:
        ``{"set": {...}, "drop": [...], "edit": {...}, "add": [...]}``, possibly empty.
    """
    rng = tile_rng(cfg.seed, int(tile["terrain_id"]), salt=0x7E44)
    ovs = list(tile.get("overlays", []))
    sx, sy = float(tile["size_x"]), float(tile["size_y"])
    out: dict = {"set": {}, "drop": [], "edit": {}, "add": []}
    if not ovs:
        return out

    steps = [i for i in walkable(ovs) if is_step(ovs[i])]
    plats = [i for i in walkable(ovs) if not is_step(ovs[i])]
    floor_top = flush_floor_z(ovs, feet, bool(tile.get('has_floor', True)))

    # --- decoration: strip first, so a wall's presence is this variant's decision -------
    if cfg.strip_existing_decor:
        out["drop"].extend(i for i, o in enumerate(ovs) if o.get("decor"))

    # --- floor ---------------------------------------------------------------------------
    mode = str(rng.choice(cfg.floor_modes))
    out["set"]["variant_floor"] = mode
    if mode == "flush" and floor_top > 0.0:
        # The WHOLE tile footprint, topped at the surface the motion starts from. Where the
        # tile already has a floor this fills the plinth; where it has none it creates ground
        # at that height. Either way no walkable surface MOVES, which is what keeps every
        # recorded pose valid.
        #
        # Exactly the footprint, and deliberately not a margin past it: verified over the
        # library, no walkable overlay and no recorded foot lies outside its tile, so there
        # is nothing to catch by overhanging -- and tiles are laid out with gaps between
        # them, so ground spilling into a gap would render as a bridge to the neighbour.
        # That is a new spurious cue, which is the thing this whole pass exists to remove.
        out["add"].append(_box(sx / 2.0, sy / 2.0, 0.0, sx, sy, floor_top, decor=False))
        out["set"]["has_floor"] = True
    elif mode == "none":
        out["set"]["has_floor"] = False

    # --- width: widen the flight only ------------------------------------------------------
    if steps:
        k = float(rng.uniform(*cfg.widen_range))
        if k > 1.001:
            for i in steps:
                w = float(ovs[i]["size_xyz"][1])
                out["edit"].setdefault(str(i), {})["size_xyz"] = [
                    float(ovs[i]["size_xyz"][0]), min(w * k, sy), float(ovs[i]["size_xyz"][2])]

    # --- truncate the landing platform ------------------------------------------------------
    if plats and feet.size:
        far = max(plats, key=lambda i: aabb_xy(ovs[i])[2])
        x0, y0, x1, y1 = aabb_xy(ovs[far])
        on = feet_on(feet, ovs[far])
        reach = float(feet[on, 0].max()) if on.any() else x0
        frac = float(rng.uniform(*cfg.truncate_frac_range))
        # Never inside where a foot went: the cut starts at the furthest one, plus a stride.
        cut = max(x0 + (x1 - x0) * frac, reach + 0.35)
        if cut < x1 - 0.05:
            out["edit"].setdefault(str(far), {})["size_xyz"] = [
                cut - x0, float(ovs[far]["size_xyz"][1]), float(ovs[far]["size_xyz"][2])]
            out["edit"][str(far)]["pos_xy"] = [(x0 + cut) / 2.0, float(ovs[far]["pos_xy"][1])]
            out["set"]["variant_platform_end"] = round(cut, 3)

    # --- walls, close in --------------------------------------------------------------------
    wmode = str(rng.choice(cfg.wall_modes))
    out["set"]["variant_walls"] = wmode
    if wmode != "none" and steps:
        ys = [aabb_xy(ovs[i]) for i in steps]
        fx0 = min(b[0] for b in ys); fx1 = max(b[2] for b in ys)
        fy0 = min(b[1] for b in ys); fy1 = max(b[3] for b in ys)
        # Along the flight, past whatever widening was just asked for.
        wid = max((out["edit"].get(str(i), {}).get("size_xyz", ovs[i]["size_xyz"])[1]
                   for i in steps), default=fy1 - fy0)
        cy = (fy0 + fy1) / 2.0
        h = float(rng.uniform(1.2, 2.4))
        length = (fx1 - fx0) + 2.0
        if wmode in ("one", "both"):
            out["add"].append(_box((fx0 + fx1) / 2.0, cy + wid / 2.0 + WALL_GAP_M, 0.0,
                                   length, WALL_T_M, h, decor=True, cls="wall_side"))
        if wmode == "both":
            out["add"].append(_box((fx0 + fx1) / 2.0, cy - wid / 2.0 - WALL_GAP_M, 0.0,
                                   length, WALL_T_M, h, decor=True, cls="wall_side"))
        if wmode == "end":
            # Past the platform, wherever it now ends: the truncation above may have moved
            # it, and a tile with no platform at all has none to stand at.
            end_x = out["set"].get("variant_platform_end")
            if end_x is None:
                end_x = max((aabb_xy(ovs[i])[2] for i in plats), default=None)
            if end_x is not None:
                out["add"].append(_box(float(end_x) + WALL_GAP_M, cy, 0.0,
                                       WALL_T_M, max(wid + 1.0, 2.0), h,
                                       decor=True, cls="wall_end"))
    return out


def apply_edits(overlays: list[dict], block: dict) -> list[dict]:
    """The overlay list an edit block produces. Pure; the input is not touched.

    Order is fixed and matters: EDIT by original index, then DROP by original index, then
    ADD. Doing it any other way makes an index mean two different things inside one block.

    Args:
        overlays: The tile's overlays.
        block: A schema-2 per-tile block.

    Returns:
        The resulting overlay list.
    """
    out = []
    drop = {int(i) for i in block.get("drop", [])}
    edit = {int(k): v for k, v in (block.get("edit") or {}).items()}
    for i, o in enumerate(overlays):
        if i in drop:
            continue
        o = dict(o)
        o.update(edit.get(i, {}))
        out.append(o)
    return out + [dict(o) for o in block.get("add", [])]


def check_feet(tile: dict, block: dict, feet: np.ndarray,
               tol: float = FOOT_TOL_M) -> list[str]:
    """Why this edit is inadmissible, or an empty list.

    The clutter pass vetoed a box that could be walked INTO. This is the mirror image: these
    edits take ground away, so what has to be proved is that nothing the robot stood on
    stopped existing, and that nothing it stood on is now buried.

    Both directions are checked, because a raised floor fails the second while passing the
    first -- a foot recorded on the old ground at z = 0 is inside a slab whose top is 0.13.

    Args:
        tile: The manifest entry.
        block: The proposed edit block.
        feet: ``(N, 3)`` tile-local recorded foot positions.
        tol: Vertical slack on "standing on".

    Returns:
        Human-readable reasons; empty means admissible.
    """
    if feet.size == 0:
        return []
    before = list(tile.get("overlays", []))
    after = apply_edits(before, block)
    sets = block.get("set", {})
    had_floor = bool(tile.get("has_floor", True))
    now_floor = bool(sets.get("has_floor", had_floor))
    bad: list[str] = []

    # 1. Every foot still has something under it. "Under" = an overlay whose top is at the
    #    foot, or the tile floor when the tile has one.
    def supported(fset: list[dict], floor: bool) -> np.ndarray:
        sup = np.zeros(len(feet), dtype=bool)
        for o in fset:
            if o.get("decor"):
                continue
            sup |= feet_on(feet, o, tol)
        if floor:
            sup |= np.abs(feet[:, 2]) <= tol          # the tile's own floor sits at z = 0
        return sup

    lost = supported(before, had_floor) & ~supported(after, now_floor)
    if lost.any():
        bad.append(f"{int(lost.sum())} recorded foot sample(s) lose the ground they stood on")

    # 2. No foot is left INSIDE a surface. Only added/raised geometry can do this, so it is
    #    checked against what the edit introduces rather than against everything.
    for o in block.get("add", []):
        if o.get("decor"):
            continue
        x0, y0, x1, y1 = aabb_xy(o)
        inside = ((feet[:, 0] >= x0) & (feet[:, 0] <= x1)
                  & (feet[:, 1] >= y0) & (feet[:, 1] <= y1)
                  & (feet[:, 2] < top_z(o) - tol))
        if inside.any():
            bad.append(f"{int(inside.sum())} foot sample(s) end up buried under a new "
                       f"surface at z={top_z(o):.3f}")

    # 3. A decoration must not stand where the robot did -- it cannot be collided with, but a
    #    wall drawn through the robot is a picture of something that never happened.
    for o in block.get("add", []):
        if not o.get("decor"):
            continue
        x0, y0, x1, y1 = aabb_xy(o)
        hit = ((feet[:, 0] >= x0 - 0.20) & (feet[:, 0] <= x1 + 0.20)
               & (feet[:, 1] >= y0 - 0.20) & (feet[:, 1] <= y1 + 0.20))
        if hit.any():
            bad.append(f"{int(hit.sum())} foot sample(s) fall inside decoration "
                       f"'{o.get('variant_class', 'decor')}'")
    return bad


def plan_tile(tile: dict, feet: np.ndarray, cfg: VariantCfg,
              tries: int = 8) -> tuple[dict, list[str]]:
    """Draw an edit block for one tile, backing off until it is admissible.

    Rejecting a tile outright would quietly bias the variant set toward the tiles that
    happen to be easy -- and the hard ones are the stair transitions, which are the whole
    point. So a failed draw is retried with a progressively more conservative config rather
    than dropped, and the identity block (which changes nothing and is admissible by
    construction) is the floor.

    The two things that fail in practice, both caught by :func:`check_feet`:

    * a FLUSH floor buries feet that were recorded on the ground BESIDE the approach
      platform -- exactly the plinth the flush floor exists to remove, stepped off;
    * removing the floor strands feet that were standing on it.

    Both are properties of where the robot actually went on that tile, so they are decided
    per tile rather than per family.

    Args:
        tile: The manifest entry.
        feet: ``(N, 3)`` tile-local recorded foot positions.
        cfg: What the variant may do.
        tries: How many progressively-safer draws to attempt.

    Returns:
        ``(block, reasons)``. ``reasons`` is why the earlier draws were refused, kept so a
        caller can report what the terrain would not allow rather than only what it did.
    """
    history: list[str] = []
    for k in range(tries):
        # Back off in the order the failures appear: first drop the floor edit, then the
        # truncation, then the widening -- decoration is last because it is visual-only and
        # is never the reason a foot loses its ground.
        c = VariantCfg(
            seed=cfg.seed + 1000 * k, strip_existing_decor=cfg.strip_existing_decor,
            floor_modes=cfg.floor_modes if k < 2 else ("keep",),
            widen_range=cfg.widen_range if k < 6 else (1.0, 1.0),
            truncate_frac_range=cfg.truncate_frac_range if k < 4 else (1.0, 1.0),
            wall_modes=cfg.wall_modes,
        )
        blk = plan_variant(tile, feet, c)
        why = check_feet(tile, blk, feet)
        if not why:
            return blk, history
        history.extend(why)
    # Identity: strip the decoration (always safe) and nothing else.
    blk = {"set": {"variant_floor": "keep", "variant_walls": "none"},
           "drop": [i for i, o in enumerate(tile.get("overlays", [])) if o.get("decor")]
           if cfg.strip_existing_decor else [],
           "edit": {}, "add": []}
    return blk, history


SCHEMA = 2


def merge_variants(manifest: list[dict], doc: dict, source: str | None = None) -> int:
    """Apply a schema-2 variant sidecar to a manifest, IN PLACE. Returns tiles changed.

    The clutter sidecar (schema 1) only ever APPENDED decoration, so merging it was one
    ``extend``. This one edits and removes real geometry, so it carries four operations and
    they are applied in a fixed order -- edit, drop, add -- because an index has to mean the
    same thing throughout one block.

    Args:
        manifest: The library's tile list; mutated.
        doc: The parsed sidecar.
        source: The library's filename, checked against the sidecar's if given.

    Returns:
        How many tiles were changed.

    Raises:
        ValueError: On the wrong schema, a source mismatch, or an unknown terrain_id.
    """
    if int(doc.get("schema", 0)) != SCHEMA:
        raise ValueError(f"terrain-variant sidecar schema {doc.get('schema')!r} is not {SCHEMA}")
    if source is not None and doc.get("source") and doc["source"] != source:
        raise ValueError(
            f"variant sidecar was authored against {doc['source']!r}, not {source!r}. Its "
            "edits are BY INDEX into that library's overlays, so applying it to another "
            "reshapes whichever tile happens to share an id.")
    by_id = {int(t["terrain_id"]): t for t in manifest if "terrain_id" in t}
    unknown = [k for k in doc["per_terrain_id"] if int(k) not in by_id]
    if unknown:
        raise ValueError(f"variant names terrain_ids absent from the manifest: {unknown[:8]}")
    changed = 0
    for k, blk in doc["per_terrain_id"].items():
        tile = by_id[int(k)]
        tile["overlays"] = apply_edits(list(tile.get("overlays", [])), blk)
        for field_name, value in (blk.get("set") or {}).items():
            tile[field_name] = value
        changed += 1
    return changed
