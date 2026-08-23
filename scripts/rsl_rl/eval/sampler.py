# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tile/clip selection, tile-column order, and env allocation. IsaacLab-free.

Everything here is a pure function of (library content, arguments), so the whole selection
is reproducible from the recorded plan and unit-testable without a simulator.

Three facts about the library drive the design, all measured rather than assumed:

* The useful stratum is the manifest's ``family`` string (8 of them), not
  ``terrain_type_code`` (5 values) -- ``box`` alone spans distractors, jump-on and two
  jump-offs, which fail in visibly different ways.
* **Flat is not tiled.** Every flat clip carries ``terrain_id == -1``; the flat tiles are
  SYNTHESISED by the importer, sized from ``generator_config.counts.flat``, and any flat
  clip may land on any flat tile.  So flat is selected clip-wise and its tile count is a
  separate knob.
* Clip lengths differ 3.3x across families (300 frames for a distractor, 999 for flat), so
  an even split of envs over tiles leaves the flat lane running long after every other lane
  has drained -- measured at 1.97x the balanced wall clock.  Hence :func:`allocate_envs`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import numpy as np

#: ``terrain_id`` of every flat clip.  Mirrors ``motion_gen.clips.clip_io._FLAT_TERRAIN_ID``;
#: duplicated as a named constant because this module deliberately reads the ``.h5`` with
#: h5py alone rather than importing the loader (which pulls torch).
FLAT_TERRAIN_ID: int = -1

#: Tile class order the importer appends specs in, and therefore the order of tile COLUMNS
#: in the assembled terrain.  Mirrors ``clip_terrain_loader.assemble_clip_terrain``; pinned
#: against that function's own output by a test, so a change there fails loudly here.
TERRAIN_CLASS_ORDER: tuple[str, ...] = ("terrain", "transition", "slope", "box")

TILE_PICKS: tuple[str, ...] = ("spread", "random", "first")
CLIP_PICKS: tuple[str, ...] = ("spread", "longest", "random")


def normalize_family(raw: str) -> str:
    """Strip the manifest's ``family_`` prefix and ``.h5`` suffix.

    The manifest stores the generating file name (``family_stairs_up.h5``); every consumer
    wants the bare label.  Normalising once here beats nine call sites doing it.

    Args:
        raw: The manifest's ``family`` value.

    Returns:
        The bare family name, e.g. ``stairs_up``.
    """
    name = str(raw)
    if name.startswith("family_"):
        name = name[len("family_"):]
    if name.endswith(".h5"):
        name = name[: -len(".h5")]
    return name


@dataclass(frozen=True)
class ClipIndex:
    """The per-clip selection columns of a library, plus its terrain metadata.

    Read with h5py only -- no torch, no IsaacLab -- so the plan stage stays light and can
    run anywhere.  Deliberately does NOT hold ``y_des``/``dy_des``: selection never needs
    the motion itself, and those are gigabytes.
    """

    path: str
    terrain_id: np.ndarray          # [N] int64, FLAT_TERRAIN_ID for flat
    terrain_type_code: np.ndarray   # [N] int64
    clip_len: np.ndarray            # [N] int64, real frames before repeat-last padding
    anchor_mask: np.ndarray         # [N] int64, per-clip root-DOF FREE/FIXED bitmask
    manifest: list[dict]            # one entry per NON-FLAT tile
    terrain_type_names: list[str]
    num_flat_tiles_source: int      # generator_config.counts.flat as found
    dt: float
    num_steps: int

    @property
    def num_clips(self) -> int:
        """Number of clips in the library."""
        return int(self.terrain_id.size)

    @property
    def family_of_tile(self) -> dict[int, str]:
        """``terrain_id`` -> normalised family name, for non-flat tiles."""
        return {
            int(m["terrain_id"]): normalize_family(m.get("family", m.get("type", "?")))
            for m in self.manifest
        }

    @property
    def type_of_tile(self) -> dict[int, str]:
        """``terrain_id`` -> manifest ``type`` (the coarse class), for non-flat tiles."""
        return {int(m["terrain_id"]): str(m["type"]) for m in self.manifest}

    def tiles_by_family(self) -> dict[str, list[int]]:
        """Sorted ``terrain_id`` list per family, restricted to tiles that HAVE clips.

        A manifest tile with no clip would be built as terrain and then have nothing to
        spawn on it, which the terrain reset refuses at runtime; excluding it here turns
        that into a selection-time non-event.
        """
        have = {int(t) for t in np.unique(self.terrain_id) if int(t) != FLAT_TERRAIN_ID}
        out: dict[str, list[int]] = {}
        for tile, fam in self.family_of_tile.items():
            if tile in have:
                out.setdefault(fam, []).append(tile)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def clips_on_tile(self, tile: int) -> np.ndarray:
        """Ascending clip indices standing on ``tile`` (``FLAT_TERRAIN_ID`` = the flat pool)."""
        return np.flatnonzero(self.terrain_id == int(tile))


def read_clip_index(lib_path: str) -> ClipIndex:
    """Read a clip library's selection columns and terrain metadata.

    Args:
        lib_path: Path to the clip library ``.h5``.

    Returns:
        The populated :class:`ClipIndex`.

    Raises:
        KeyError: If the library lacks ``terrain_id`` (i.e. predates terrain support), which
            makes every stratum in this module undefined.
    """
    import h5py  # noqa: PLC0415 - keeps module import cheap for pure-arithmetic callers

    with h5py.File(str(lib_path), "r") as f:
        if "terrain_id" not in f:
            raise KeyError(
                f"{lib_path} has no 'terrain_id' dataset, so it predates terrain clips and "
                "cannot be stratified by tile or family."
            )
        terrain_id = f["terrain_id"][:].astype(np.int64)
        type_code = f["terrain_type_code"][:].astype(np.int64)
        clip_len = f["clip_len"][:].astype(np.int64)
        anchor_mask = f["anchor_mask"][:].astype(np.int64)

        def _attr(name: str, default):
            raw = f.attrs.get(name)
            if raw is None:
                return default
            return json.loads(raw) if isinstance(raw, (str, bytes)) else raw

        manifest = _attr("terrain_manifest", [])
        type_names = _attr("terrain_type_names", [])
        gen_cfg = _attr("generator_config", {}) or {}
        counts = gen_cfg.get("counts") or {} if isinstance(gen_cfg, dict) else {}
        n_flat_clips = int((terrain_id == FLAT_TERRAIN_ID).sum())
        num_flat = int(counts.get("flat", n_flat_clips))
        dt = float(f.attrs.get("dt", 0.02))
        num_steps = int(f.attrs.get("num_steps", clip_len.max() if clip_len.size else 0))

    return ClipIndex(
        path=str(lib_path), terrain_id=terrain_id, terrain_type_code=type_code,
        clip_len=clip_len, anchor_mask=anchor_mask, manifest=list(manifest),
        terrain_type_names=list(type_names), num_flat_tiles_source=num_flat,
        dt=dt, num_steps=num_steps,
    )


def tile_column_order(manifest: list[dict], num_flat_tiles: int) -> list[int]:
    """The ``terrain_id`` of each tile COLUMN, in the order the importer builds them.

    Reproduces ``clip_terrain_loader.assemble_clip_terrain``: the manifest is sorted by
    ``terrain_id``, specs are appended in :data:`TERRAIN_CLASS_ORDER` (ascending
    ``terrain_id`` within each class), and ``num_flat_tiles`` flat columns follow.

    Everything downstream leans on this: the pin list, the env allocation and the coverage
    proof are all indexed by column, so if this order were wrong the eval would silently
    pair each env with a clip belonging to a different tile.  A test pins it against
    ``assemble_clip_terrain``'s own output rather than against a hand-written list alone.

    Args:
        manifest: The library's ``terrain_manifest`` (non-flat tiles only).
        num_flat_tiles: How many flat columns the importer will synthesise.

    Returns:
        One ``terrain_id`` per column; ``FLAT_TERRAIN_ID`` for the flat columns.

    Raises:
        ValueError: If a manifest entry carries a ``type`` outside
            :data:`TERRAIN_CLASS_ORDER`, which would mean the importer's class list and
            this one have diverged.
    """
    by_type: dict[str, list[int]] = {c: [] for c in TERRAIN_CLASS_ORDER}
    for m in sorted(manifest, key=lambda e: int(e["terrain_id"])):
        ttype = str(m["type"])
        if ttype not in by_type:
            raise ValueError(
                f"terrain_manifest entry {m.get('terrain_id')} has type {ttype!r}, which is "
                f"not in TERRAIN_CLASS_ORDER {TERRAIN_CLASS_ORDER}. The importer's class "
                "ordering has changed; update TERRAIN_CLASS_ORDER to match it."
            )
        by_type[ttype].append(int(m["terrain_id"]))
    cols: list[int] = []
    for cls in TERRAIN_CLASS_ORDER:
        cols.extend(by_type[cls])
    cols.extend([FLAT_TERRAIN_ID] * int(num_flat_tiles))
    return cols


def _pick_spread(items: np.ndarray, k: int) -> np.ndarray:
    """``k`` evenly spaced entries of a sorted array, without duplicates.

    Systematic rather than random: the library writes clips grouped by family and tile, so
    a prefix would be a handful of tiles and a random draw can clump.  Evenly spaced spans
    the generation order, which is the same reason ``make_flat_clip_lib`` uses it.
    """
    if k >= items.size:
        return items
    idx = np.unique(np.linspace(0, items.size - 1, k).round().astype(np.int64))
    return items[idx]


@dataclass(frozen=True)
class SelectedClip:
    """One clip in the plan, with everything selection knew about it."""

    source_clip_id: int   # index in the SOURCE library
    terrain_id: int       # FLAT_TERRAIN_ID for flat
    family: str           # normalised; "flat" for flat clips
    clip_len: int
    column: int           # index into the plan's tile_columns


def select_clips(
    index: ClipIndex,
    *,
    tiles_per_family: int | None = None,
    clips_per_tile: int = 2,
    flat_tiles: int = 12,
    flat_clips: int | None = None,
    min_clip_len: int = 52,
    families: tuple[str, ...] | None = None,
    exclude_families: tuple[str, ...] = (),
    tile_pick: str = "spread",
    clip_pick: str = "spread",
    seed: int = 20260820,
) -> tuple[list[SelectedClip], list[int]]:
    """Choose a stratified clip set and the tile columns it will run on.

    Selection order is fixed (families sorted by name, tiles ascending, clips ascending) and
    a single seeded generator is consumed in that order, so the result is a pure function of
    (library content, arguments).

    The returned clips are ordered by tile COLUMN, which is what lets the pin list, the env
    allocation and the coverage proof all index by column without a lookup table.

    Args:
        index: The source library's :class:`ClipIndex`.
        tiles_per_family: Tiles to take from each family; ``None`` takes every tile the
            family has, which maximises the number of bootstrap clusters and is the default
            because tile count -- not robots per clip -- is what sets the interval width.
        clips_per_tile: Clips per chosen tile.
        flat_tiles: Flat COLUMNS to build.  Defaults to 12 because ``flat_tile_decor`` holds
            12 room variants handed round-robin per flat tile, and the decor is in the depth
            cameras' target list -- fewer tiles silently drops variants from what the policy
            sees.
        flat_clips: Flat clips to draw; ``None`` means ``flat_tiles * clips_per_tile``.
        min_clip_len: Reject clips shorter than this.  The default 52 is
            ``max(REFERENCE_OFFSETS) + 2``: at or below the observation window a clip has no
            unsaturated step and ``clip_end_reached`` would truncate it at step 0.
        families: Restrict to these families (normalised names); ``None`` takes all.
        exclude_families: Drop these families after the ``families`` filter.  Applies to the
            TERRAIN families only -- flat is not one of them, and is controlled by
            ``flat_tiles``.
        tile_pick: One of :data:`TILE_PICKS`.
        clip_pick: One of :data:`CLIP_PICKS`.
        seed: Seed for the ``random`` pick modes.

    Returns:
        ``(clips, tile_columns)`` -- the selection in column order, and the ``terrain_id``
        per column.

    Raises:
        ValueError: On an unknown pick mode, a requested family the library lacks, a family
            with fewer tiles than requested, a tile that cannot supply ``clips_per_tile``
            long-enough clips, or a flat request the library cannot fill.
    """
    if tile_pick not in TILE_PICKS:
        raise ValueError(f"tile_pick must be one of {TILE_PICKS}, got {tile_pick!r}")
    if clip_pick not in CLIP_PICKS:
        raise ValueError(f"clip_pick must be one of {CLIP_PICKS}, got {clip_pick!r}")
    if clips_per_tile < 1:
        raise ValueError(f"clips_per_tile must be >= 1, got {clips_per_tile}")
    if flat_tiles < 0:
        raise ValueError(f"flat_tiles must be >= 0, got {flat_tiles}")

    rng = np.random.default_rng(seed)
    by_family = index.tiles_by_family()
    if families is not None:
        wanted = [normalize_family(f) for f in families]
        missing = [f for f in wanted if f not in by_family]
        if missing:
            raise ValueError(
                f"library {index.path} has no tiles for family/families {missing}; it has "
                f"{sorted(by_family)}"
            )
        by_family = {f: by_family[f] for f in sorted(wanted)}
    if exclude_families:
        drop = {normalize_family(f) for f in exclude_families}
        unknown = drop - set(by_family)
        if unknown:
            raise ValueError(
                f"cannot exclude family/families {sorted(unknown)}: the library has "
                f"{sorted(by_family)}. (Flat is excluded with --flat_tiles 0, not here.)"
            )
        by_family = {f: v for f, v in by_family.items() if f not in drop}
        if not by_family:
            raise ValueError("every terrain family was excluded; nothing to evaluate")

    # --- choose tiles, family by family -------------------------------------------------
    chosen_tiles: dict[str, list[int]] = {}
    for fam in sorted(by_family):
        tiles = np.asarray(by_family[fam], dtype=np.int64)
        k = tiles.size if tiles_per_family is None else int(tiles_per_family)
        if k > tiles.size:
            raise ValueError(
                f"family {fam!r} has {tiles.size} tiles but {k} were requested. Lower "
                "--tiles_per_family, or pass None to take every tile."
            )
        if tile_pick == "spread":
            picked = _pick_spread(tiles, k)
        elif tile_pick == "first":
            picked = tiles[:k]
        else:
            picked = np.sort(rng.choice(tiles, size=k, replace=False))
        chosen_tiles[fam] = [int(t) for t in picked]

    # --- choose clips on each chosen tile ------------------------------------------------
    clips_on_column: dict[int, list[SelectedClip]] = {}
    for fam in sorted(chosen_tiles):
        for tile in chosen_tiles[fam]:
            pool = index.clips_on_tile(tile)
            lens = index.clip_len[pool]
            ok = pool[lens >= int(min_clip_len)]
            if ok.size < clips_per_tile:
                raise ValueError(
                    f"tile {tile} (family {fam!r}) has only {ok.size} clips with "
                    f"clip_len >= {min_clip_len}, but {clips_per_tile} were requested."
                )
            if clip_pick == "spread":
                picked = _pick_spread(ok, clips_per_tile)
            elif clip_pick == "longest":
                picked = np.sort(ok[np.argsort(-index.clip_len[ok], kind="stable")][:clips_per_tile])
            else:
                picked = np.sort(rng.choice(ok, size=clips_per_tile, replace=False))
            clips_on_column[tile] = [
                SelectedClip(int(c), int(tile), fam, int(index.clip_len[c]), -1) for c in picked
            ]

    # --- flat, which has no tiles of its own ---------------------------------------------
    n_flat = int(flat_tiles * clips_per_tile if flat_clips is None else flat_clips)
    flat_selected: list[SelectedClip] = []
    if flat_tiles > 0 and n_flat > 0:
        pool = index.clips_on_tile(FLAT_TERRAIN_ID)
        ok = pool[index.clip_len[pool] >= int(min_clip_len)]
        if ok.size < n_flat:
            raise ValueError(
                f"library {index.path} has {ok.size} flat clips with clip_len >= "
                f"{min_clip_len}, but {n_flat} were requested."
            )
        picked = _pick_spread(ok, n_flat) if clip_pick != "random" else np.sort(
            rng.choice(ok, size=n_flat, replace=False))
        flat_selected = [
            SelectedClip(int(c), FLAT_TERRAIN_ID, "flat", int(index.clip_len[c]), -1)
            for c in picked
        ]

    # --- lay them out in tile-column order ----------------------------------------------
    kept_manifest = [m for m in index.manifest if int(m["terrain_id"]) in clips_on_column]
    columns = tile_column_order(kept_manifest, flat_tiles)
    out: list[SelectedClip] = []
    flat_cols = [i for i, t in enumerate(columns) if t == FLAT_TERRAIN_ID]
    for col, tid in enumerate(columns):
        if tid == FLAT_TERRAIN_ID:
            continue
        for sc in clips_on_column[tid]:
            out.append(dataclasses.replace(sc, column=col))
    # Flat clips pool across flat columns -- any flat clip is legal on any flat tile -- so
    # deal them round-robin rather than binding a clip to a column.
    for i, sc in enumerate(flat_selected):
        out.append(dataclasses.replace(sc, column=flat_cols[i % len(flat_cols)] if flat_cols else -1))
    # Sort so a SUBSET clip id is monotone in column. The round-robin deal above cycles
    # through the flat columns, so without this the flat tail is not column-ordered and
    # "clip id k lives on column c, and c only grows" stops holding -- which is the property
    # the pin list and the coverage proof are written against.
    out.sort(key=lambda sc: (sc.column, sc.source_clip_id))
    return out, columns


def allocate_envs(weights: np.ndarray, num_envs: int) -> np.ndarray:
    """Split ``num_envs`` across columns in proportion to ``weights``, exactly.

    Largest-remainder (Hare-Niemeyer): every column gets one env first, then the remainder
    is handed out by the largest fractional share, ties broken by column index.  The
    guarantee that matters is ``result.sum() == num_envs`` -- naive rounding of the
    proportional shares overshot by 4 envs on a 1440-env request.

    Weighting by the column's own expected work (clips x robots x mean clip length) is what
    keeps the 20 s flat lane from setting the wall clock: an even split measured 1.97x the
    balanced time.

    Args:
        weights: Non-negative per-column weight; higher means more envs.
        num_envs: Total envs to distribute.  Must be at least one per column, since a
            column with no env would silently drop a whole family from the eval.

    Returns:
        ``[num_columns]`` int64 counts, each >= 1, summing to ``num_envs``.

    Raises:
        ValueError: If there are no columns, if ``num_envs`` is below the column count, or
            if the weights are negative or sum to zero.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = w.size
    if n == 0:
        raise ValueError("allocate_envs needs at least one column")
    if np.any(w < 0):
        raise ValueError("allocate_envs weights must be non-negative")
    if w.sum() <= 0:
        raise ValueError("allocate_envs weights sum to zero, so no split is defined")
    if num_envs < n:
        raise ValueError(
            f"num_envs={num_envs} is fewer than the {n} tile columns in the plan. Every "
            "column needs at least one env or its family drops out of the eval entirely; "
            "raise num_envs or lower --tiles_per_family / --flat_tiles."
        )
    extra = int(num_envs) - n
    share = w / w.sum() * extra
    base = np.floor(share).astype(np.int64)
    leftover = extra - int(base.sum())
    if leftover > 0:
        frac = share - base
        # -frac for descending; arange breaks ties by column index, stable under lexsort.
        order = np.lexsort((np.arange(n), -frac))
        base[order[:leftover]] += 1
    counts = base + 1
    assert counts.sum() == num_envs, (counts.sum(), num_envs)  # invariant, cheap to hold
    return counts


def columns_of_env(counts: np.ndarray) -> np.ndarray:
    """Expand per-column env counts into a ``[num_envs]`` column index per env.

    Contiguous by column, matching the shape of IsaacLab's own uniform mapping
    (``terrain_types = floor(arange(N) / (N/num_cols))``) so a reader comparing the two sees
    the same structure with different widths.
    """
    counts = np.asarray(counts, dtype=np.int64)
    return np.repeat(np.arange(counts.size, dtype=np.int64), counts)


def assign_pins(clips: list[SelectedClip], env_columns: np.ndarray) -> list[int]:
    """Pin each env to one clip standing on that env's own column.

    Returned list is exactly ``num_envs`` long, which is the point: ``resolve_pinned_clips``
    maps ``ids[env_id % len(ids)]``, so at that length the list IS the per-env assignment and
    "give env e clip k" is ``pin[e] = k`` -- no arithmetic in the reset function.

    Envs on a column are dealt its clips round-robin, so each clip of a column gets as equal
    a share of that column's envs as the counts allow.

    Args:
        clips: The plan's clips, whose position in the list is the SUBSET clip id.
        env_columns: ``[num_envs]`` column index per env, from :func:`columns_of_env`.

    Returns:
        ``[num_envs]`` subset clip ids.

    Raises:
        ValueError: If any column carrying envs has no clips (which the terrain reset would
            later refuse with a much less specific message).
    """
    by_col: dict[int, list[int]] = {}
    for subset_id, sc in enumerate(clips):
        by_col.setdefault(int(sc.column), []).append(subset_id)
    pins: list[int] = []
    seen: dict[int, int] = {}
    for env_id, col in enumerate(np.asarray(env_columns, dtype=np.int64).tolist()):
        options = by_col.get(int(col))
        if not options:
            raise ValueError(
                f"env {env_id} sits on column {col}, which has no clips in the plan. The "
                "terrain reset validates that a pinned clip's terrain_id matches its env's "
                "tile, so this would abort at the first reset."
            )
        k = seen.get(col, 0)
        pins.append(options[k % len(options)])
        seen[col] = k + 1
    return pins


class JobQueue:
    """Work-conserving dispatcher: which clip each env runs next, and when the cell is done.

    A cell owes every clip ``robots_per_clip`` completed episodes.  A *static* env->clip
    pinning cannot deliver that efficiently, for two reasons measured on the real plan:

    * **Granularity.** With 152 columns and 256 envs each column gets 1-3 envs, while the
      work per column spans 3.3x (a 999-frame flat clip against a 300-frame distractor).
      The slowest column then sets the wall clock: 1.93 h against 1.02 h if the same total
      work were spread evenly.
    * **The env-count floor.** Static pinning needs at least one env per column, so "every
      tile in the library" would demand >= 152 envs whether or not that fits in memory.

    Dispatching on demand fixes both: an env that finishes takes the clip with the most
    outstanding work, so no env idles while work remains, and a single env can visit every
    column in turn.  It prefers a clip on the env's CURRENT column when one still has work,
    which keeps the common case free of tile changes without giving up work conservation.

    Episodes are counted on ASSIGNMENT, not completion, so the queue drains monotonically;
    episodes still in flight when the rollout hits its step ceiling are reported by
    :meth:`stats` and must be written as administratively censored rather than dropped.
    """

    def __init__(
        self,
        clips: list[SelectedClip],
        robots_per_clip: int,
        num_envs: int,
        *,
        initial_columns: np.ndarray | None = None,
    ) -> None:
        """Build the queue.

        Args:
            clips: The plan's clips; list position is the subset clip id.
            robots_per_clip: Episodes owed per clip.
            num_envs: Number of parallel envs.
            initial_columns: Optional ``[num_envs]`` starting column per env; defaults to a
                work-proportional allocation.  Only sets where each env STARTS looking --
                dispatch may move it once its column is exhausted.

        Raises:
            ValueError: If ``robots_per_clip`` or ``num_envs`` is below 1, or if there are
                no clips.
        """
        if not clips:
            raise ValueError("JobQueue needs at least one clip")
        if int(robots_per_clip) < 1:
            raise ValueError(f"robots_per_clip must be >= 1, got {robots_per_clip}")
        if int(num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        self.clips = list(clips)
        self.robots_per_clip = int(robots_per_clip)
        self.num_envs = int(num_envs)
        self._len = np.array([c.clip_len for c in clips], dtype=np.float64)
        self._col = np.array([c.column for c in clips], dtype=np.int64)
        self._assigned = np.zeros(len(clips), dtype=np.int64)
        self._done = np.zeros(len(clips), dtype=np.int64)
        self.parked_env_steps = 0

        if initial_columns is None:
            weights = np.zeros(int(self._col.max()) + 1, dtype=np.float64)
            for c in clips:
                weights[c.column] += c.clip_len
            # An env count below the column count is legal here (unlike static pinning), so
            # only allocate proportionally when there is room for one env per column.
            if self.num_envs >= weights.size:
                initial_columns = columns_of_env(allocate_envs(weights, self.num_envs))
            else:
                order = np.argsort(-weights, kind="stable")
                initial_columns = order[: self.num_envs].astype(np.int64)
        self.env_column = np.asarray(initial_columns, dtype=np.int64).copy()

        # The first assignment goes through the SAME dispatch as every later one, so the
        # per-clip cap holds by construction. Dealing a column's clips round-robin over its
        # envs instead over-assigns whenever a column has more envs than it owes episodes
        # (measured: 764 episodes issued against 736 owed at 512 envs and R=4).
        self.env_clip = np.full(self.num_envs, -1, dtype=np.int64)
        self.env_active = np.zeros(self.num_envs, dtype=bool)
        for env_id in range(self.num_envs):
            self.next_job(env_id)

    def _remaining_work(self) -> np.ndarray:
        """Outstanding episodes x clip length, per clip."""
        return (self.robots_per_clip - self._assigned).clip(min=0) * self._len

    def next_job(self, env_id: int) -> int | None:
        """Clip for ``env_id`` to run next, or ``None`` when nothing is left.

        Prefers a clip on the env's current column, so an env changes tile only once its
        own column is exhausted.

        Args:
            env_id: The env that just finished an episode.

        Returns:
            A subset clip id, or ``None`` if every clip has been fully assigned.
        """
        work = self._remaining_work()
        if not work.any():
            self.env_active[env_id] = False
            return None
        here = np.where(self._col == self.env_column[env_id], work, 0.0)
        pick = int(np.argmax(here)) if here.any() else int(np.argmax(work))
        self._assigned[pick] += 1
        self.env_clip[env_id] = pick
        self.env_column[env_id] = self._col[pick]
        self.env_active[env_id] = True
        return pick

    def note_completed(self, clip_id: int) -> None:
        """Record that one episode of ``clip_id`` finished."""
        self._done[int(clip_id)] += 1

    @property
    def drained(self) -> bool:
        """True once every clip has its full quota of COMPLETED episodes.

        Stated in terms of completions rather than an in-flight counter: the counter has to
        be incremented by the caller on dispatch and decremented on completion, and getting
        those two out of step by one leaks partial episodes into the table (measured: 13
        spurious rows on a 104-episode cell).  Completions are observed directly, so there is
        nothing to keep in step.
        """
        return bool((self._done >= self.robots_per_clip).all())

    def stats(self) -> dict:
        """Progress summary, for the cell manifest and the progress line."""
        owed = int(self.robots_per_clip) * len(self.clips)
        return {
            "clips": len(self.clips),
            "robots_per_clip": self.robots_per_clip,
            "episodes_owed": owed,
            "episodes_assigned": int(self._assigned.sum()),
            "episodes_done": int(self._done.sum()),
            "episodes_in_flight": int(self.env_active.sum()),
            "clips_complete": int((self._done >= self.robots_per_clip).sum()),
            "clips_short": [
                {"clip_id": i, "done": int(d)}
                for i, d in enumerate(self._done.tolist())
                if d < self.robots_per_clip
            ][:20],
            "parked_env_steps": int(self.parked_env_steps),
        }
