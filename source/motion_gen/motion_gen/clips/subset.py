# Copyright 2026 Zachary Olkin. All rights reserved.

"""Cut a small ``_small`` sibling out of a clip library, for fast play / viewing.

A full training library is a few GB and takes a long time to load, which makes every
``play.py`` run painful. The play scripts default to a ``<name>_small.h5`` sibling holding a
handful of terrain tiles plus a few flat clips -- enough to watch the policy on each terrain
family without paying for 10k clips.

Streams with h5py and copies only the SELECTED clip rows, so it never materialises the parent's
multi-GB arrays. Row order and the time axis are preserved verbatim (same ``num_steps``, same
per-clip ``clip_len``), so a small clip behaves exactly as it did in the parent -- only the SET
of clips is reduced.

``terrain_id`` is REMAPPED to ``0..K-1`` and the ``terrain_manifest`` is subset and renumbered
to match, because the terrain importer builds its tile rows by iterating the manifest: leaving
the parent's sparse ids would produce tiles the clips no longer index.

Every output carries a ``derived_from`` attr (parent path + identity + selection), which
:mod:`play`-side resolution uses to detect a STALE small -- a small cut from a since-regenerated
parent describes terrain that no longer exists, and silently playing it is exactly the
wrong-terrain failure the mandatory ``--clip`` flag exists to prevent.
"""

from __future__ import annotations

import json
import os
from typing import Any

import h5py
import numpy as np

#: Suffix marking a derived subset library.
SMALL_SUFFIX = "_small"

#: ``terrain_id`` value for a clip with no terrain tile (flat ground).
_FLAT_TERRAIN_ID = -1

#: Defaults: enough to see every terrain family plus some flat walking, still a few MB.
DEFAULT_NUM_TILES = 10
DEFAULT_CLIPS_PER_TILE = 3
DEFAULT_FLAT_CLIPS = 6


def small_path_for(lib_path: str) -> str:
    """Return the ``_small`` sibling path for ``lib_path`` (``foo.h5`` -> ``foo_small.h5``)."""
    root, ext = os.path.splitext(str(lib_path))
    return f"{root}{SMALL_SUFFIX}{ext or '.h5'}"


def is_small_path(lib_path: str) -> bool:
    """True if ``lib_path`` already names a ``_small`` library."""
    return os.path.splitext(os.path.basename(str(lib_path)))[0].endswith(SMALL_SUFFIX)


def parent_identity(lib_path: str) -> dict[str, Any]:
    """Cheap identity of a library file, for staleness detection.

    Deliberately NOT a content hash: the parent is multi-GB and this is read on every play
    startup, so hashing would cost seconds per run. ``(size, mtime_ns, num_clips_written)``
    changes whenever any generator rewrites the file, which is the case that matters. It would
    not catch an in-place edit that preserved all three, which no writer here does.

    Args:
        lib_path: Path to the library.

    Returns:
        Dict with ``path`` (basename), ``size``, ``mtime_ns`` and ``num_clips``.
    """
    st = os.stat(lib_path)
    n = 0
    try:
        with h5py.File(lib_path, "r") as f:
            n = int(f.attrs.get("num_clips_written", f.attrs.get("num_clips", 0)))
    except OSError:                                     # pragma: no cover - unreadable parent
        pass
    return {
        "path": os.path.basename(str(lib_path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "num_clips": n,
    }


def select_clips(
    terrain_id: np.ndarray,
    num_tiles: int = DEFAULT_NUM_TILES,
    clips_per_tile: int = DEFAULT_CLIPS_PER_TILE,
    flat_clips: int = DEFAULT_FLAT_CLIPS,
    seed: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """Choose which clip rows to keep, and which terrain tiles they cover.

    Tiles are spread EVENLY over the sorted tile ids rather than taking the first ``num_tiles``,
    because generators lay families out in contiguous blocks (all stairs_up, then all
    stairs_down, ...) -- taking a prefix would yield a subset that is entirely one family.

    Args:
        terrain_id: ``[N]`` per-clip terrain id (``-1`` = flat).
        num_tiles: Terrain tiles to keep (``<= 0`` keeps none).
        clips_per_tile: Clips kept per tile.
        flat_clips: Flat (no-tile) clips to keep.
        seed: RNG seed for the within-tile clip pick.

    Returns:
        ``(rows, tiles)`` -- sorted kept row indices, and the kept tile ids in order.
    """
    rng = np.random.default_rng(seed)
    tid = np.asarray(terrain_id).reshape(-1)
    all_tiles = sorted({int(t) for t in tid.tolist() if int(t) != _FLAT_TERRAIN_ID})

    tiles: list[int] = []
    if num_tiles > 0 and all_tiles:
        k = min(int(num_tiles), len(all_tiles))
        # Even spread across the sorted ids -> one or more tiles from each family block.
        picks = np.linspace(0, len(all_tiles) - 1, k).round().astype(int)
        tiles = [all_tiles[i] for i in dict.fromkeys(picks.tolist())]

    rows: list[int] = []
    for t in tiles:
        cand = np.flatnonzero(tid == t)
        take = min(int(clips_per_tile), cand.size)
        if take > 0:
            rows.extend(rng.choice(cand, size=take, replace=False).tolist())

    flat_cand = np.flatnonzero(tid == _FLAT_TERRAIN_ID)
    take_flat = min(int(flat_clips), flat_cand.size)
    if take_flat > 0:
        rows.extend(rng.choice(flat_cand, size=take_flat, replace=False).tolist())

    return np.array(sorted(int(r) for r in rows), dtype=np.int64), tiles


def write_small_library(
    src_path: str,
    out_path: str | None = None,
    num_tiles: int = DEFAULT_NUM_TILES,
    clips_per_tile: int = DEFAULT_CLIPS_PER_TILE,
    flat_clips: int = DEFAULT_FLAT_CLIPS,
    seed: int = 0,
    compression: str | None = "lzf",
) -> dict[str, Any]:
    """Write a ``_small`` subset of ``src_path`` and return a summary.

    Args:
        src_path: Full library to subset.
        out_path: Output path (default: :func:`small_path_for` of ``src_path``).
        num_tiles: Terrain tiles to keep.
        clips_per_tile: Clips per kept tile.
        flat_clips: Flat clips to keep.
        seed: RNG seed for the within-tile pick.
        compression: h5py filter for the copied datasets.

    Returns:
        Summary dict: ``out``, ``num_clips``, ``num_tiles``, ``bytes``.

    Raises:
        ValueError: If ``src_path`` is already a ``_small`` library, or the selection is empty.
    """
    if is_small_path(src_path):
        raise ValueError(f"{src_path} is already a _small library; refusing to subset a subset")
    out = str(out_path or small_path_for(src_path))

    with h5py.File(src_path, "r") as src:
        n_src = int(src.attrs.get("num_clips_written", src.attrs.get("num_clips", 0)))
        tid_all = src["terrain_id"][:n_src] if "terrain_id" in src else np.full(n_src, -1, "int32")
        rows, tiles = select_clips(tid_all, num_tiles, clips_per_tile, flat_clips, seed)
        if rows.size == 0:
            raise ValueError(f"selection kept 0 clips from {src_path} (n={n_src})")

        # Old tile id -> new contiguous id, so the manifest the importer walks matches the clips.
        remap = {old: new for new, old in enumerate(tiles)}

        with h5py.File(out, "w") as dst:
            for key in src.keys():
                d = src[key]
                sub = d[rows] if d.shape[0] == n_src or d.shape[0] >= rows.max() + 1 else d[...]
                if key == "terrain_id":
                    sub = np.array([remap.get(int(v), _FLAT_TERRAIN_ID) for v in sub], dtype=d.dtype)
                dst.create_dataset(key, data=sub, dtype=d.dtype,
                                   compression=compression if sub.ndim > 1 else None)
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["num_clips"] = int(rows.size)
            dst.attrs["num_clips_written"] = int(rows.size)

            man = src.attrs.get("terrain_manifest")
            if man is not None:
                entries = json.loads(man) if isinstance(man, (str, bytes)) else man
                kept = []
                for e in entries:
                    if "terrain_id" not in e or int(e["terrain_id"]) not in remap:
                        continue
                    e2 = dict(e)
                    e2["terrain_id"] = remap[int(e["terrain_id"])]
                    kept.append(e2)
                kept.sort(key=lambda m: int(m["terrain_id"]))
                dst.attrs["terrain_manifest"] = json.dumps(kept)

            gen = src.attrs.get("generator")
            gen = json.loads(gen) if isinstance(gen, (str, bytes)) else gen
            dst.attrs["generator"] = json.dumps(f"{gen}+subset")
            dst.attrs["derived_from"] = json.dumps({
                **parent_identity(src_path),
                "num_tiles": len(tiles),
                "clips_per_tile": int(clips_per_tile),
                "flat_clips": int(flat_clips),
                "seed": int(seed),
            })

    return {"out": out, "num_clips": int(rows.size), "num_tiles": len(tiles),
            "bytes": os.path.getsize(out)}


def resolve_play_clip(clip_path: str, prefer_small: bool = True) -> tuple[str, list[str]]:
    """Pick which library a PLAY script should load, and what to tell the user about it.

    Play runs want the ``_small`` sibling: a full training library is multi-GB and slow to load,
    and for watching a policy a handful of tiles per family is enough. Training and recording
    must NOT use this -- they need the real clip distribution.

    Args:
        clip_path: The library the user asked for (``--clip``).
        prefer_small: False reproduces the old behaviour (load exactly what was asked for).

    Returns:
        ``(path_to_load, messages)``. ``messages`` are ready-to-print lines explaining any
        substitution, a missing sibling, or a STALE sibling -- callers should print all of them.
    """
    msgs: list[str] = []
    if not prefer_small:
        return clip_path, ["[INFO] --full_clip: loading the FULL library as given."]
    if is_small_path(clip_path):
        return clip_path, []                            # already a small; nothing to do
    small = small_path_for(clip_path)
    if not os.path.exists(small):
        msgs.append(f"[WARN] no {small}; loading the FULL library, which is slow. Make one with:")
        msgs.append(f"[WARN]   python scripts/rsl_rl/make_clip_subset.py --lib {clip_path}")
        return clip_path, msgs
    ok, why = check_small_is_current(small, clip_path)
    if not ok:
        msgs.append(f"[WARN] {small} is STALE: {why}.")
        msgs.append("[WARN] It describes terrain the parent no longer has. Loading the FULL "
                    "library instead; re-cut with make_clip_subset.py --force.")
        return clip_path, msgs
    if why:                                             # ok, but unverifiable
        msgs.append(f"[INFO] {small}: {why}")
    msgs.append(f"[INFO] play default: loading {small} instead of {clip_path} "
                f"(pass --full_clip for the full library).")
    return small, msgs


def check_small_is_current(small_path: str, parent_path: str) -> tuple[bool, str]:
    """Is ``small_path`` still a faithful subset of ``parent_path``?

    Args:
        small_path: The ``_small`` library.
        parent_path: The full library it should derive from.

    Returns:
        ``(ok, reason)``. ``ok`` is False when the small was cut from a DIFFERENT version of the
        parent (it then describes terrain the parent no longer has). ``reason`` is empty when ok,
        else a one-line explanation. A small with no ``derived_from`` (hand-made, or predating
        this module) returns ok with a note, since there is nothing to compare.
    """
    try:
        with h5py.File(small_path, "r") as f:
            raw = f.attrs.get("derived_from")
    except OSError as exc:                              # pragma: no cover - unreadable small
        return False, f"cannot read {small_path}: {exc}"
    if raw is None:
        return True, "no derived_from stamp (hand-made or pre-dates subset.py); not checked"
    rec = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    cur = parent_identity(parent_path)
    for key, label in (("size", "size"), ("mtime_ns", "mtime"), ("num_clips", "clip count")):
        if key in rec and rec[key] != cur[key]:
            return False, (f"cut from a different {os.path.basename(parent_path)} "
                           f"({label} {rec[key]} != {cur[key]})")
    return True, ""
