#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pick BALLAST windows: a fixed slice of the ORIGINAL pre-training set, carried at weight 1.

AWR reweights the generator toward one course. Ballast is what stops that quietly costing the
distribution it was pre-trained on: a fixed number of windows from the pre-training recording,
trained on at weight 1 alongside the fine-tune buffer, not graded, not fitted, and taking no
part in the advantage.

**The exclusion is STRUCTURAL, not a flag.** The ballast lives in its own file and only
``finetune_generator.py`` is ever pointed at it -- ``fit_value.py`` and ``label_advantage.py``
see the buffer alone. The alternative, merging ballast windows into the buffer, would need them
renumbered into a uid range no iteration can ever reach, because every join downstream is
``(episode_uid, replan_index)``; one bad offset there makes a ballast window indistinguishable
from a graded one, and nothing checks it. Separate files cannot be got wrong that way.

**The flat share is declared, because a terrain course contains none.** The fine-tune buffer is
entirely terrain windows, so drift on FLAT ground is invisible to it -- the buffer has no flat
rows for the critic to grade or the loss to see. ``ballast_flat_frac`` is the share drawn from
flat windows specifically, and the rest come from the terrain families, so the ballast is a
stated mixture rather than whatever the pre-training recording happened to be made of.

**A stratum that cannot fill its share is an ERROR.** Silently returning fewer windows would
make the declared mixture a fiction while every log line still read normally, and the whole
point of the number is that it is declared.
"""

from __future__ import annotations

import json

import h5py
import numpy as np

#: Family names that count as FLAT. Matched after stripping a trailing ``.h5``, the way
#: ``record_rollouts`` normalises them, and by exact name rather than substring: a substring
#: test on "flat" also matches nothing today but would match any future ``flat_*`` family
#: silently, which is the class of bug ``isaac_tile_course``'s FAMILY_KLASS table exists for.
FLAT_FAMILY_NAMES: frozenset[str] = frozenset({"flat", "family_flat", "unknown"})


def _family_names(f: h5py.File) -> list[str]:
    """The recording's own ``family_names`` table, or an empty list.

    Args:
        f: The open window file.

    Returns:
        ``family_names[code]``.
    """
    raw = f.attrs.get("family_names")
    if raw is None and "meta" in f.attrs:
        try:
            raw = json.loads(str(f.attrs["meta"])).get("family_names")
        except Exception:
            raw = None
    if raw is None:
        return []
    if isinstance(raw, (bytes, str)):
        try:
            raw = json.loads(raw if isinstance(raw, str) else raw.decode())
        except Exception:
            return []
    return [str(v) for v in np.asarray(raw).ravel().tolist()]


def flat_mask(path: str) -> np.ndarray:
    """``[W]`` bool: which windows of a recording are FLAT-family.

    Args:
        path: The window ``.h5``.

    Returns:
        The mask over every written window.

    Raises:
        SystemExit: If the recording carries no family table, so flat cannot be told from
            terrain. A declared flat share would then be unverifiable.
    """
    with h5py.File(path, "r") as f:
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        names = _family_names(f)
        if not names or "family_code" not in f:
            raise SystemExit(
                f"{path} carries no family table, so FLAT windows cannot be told from terrain "
                "ones. Either the recording predates the family stamp or it is a flat-only "
                "library (which legitimately has none). Set ballast_flat_frac to 0 to draw "
                "the ballast without a stratum, or point at a recording that has families.")
        code = f["family_code"][:n]
    norm = [v[:-3] if v.endswith(".h5") else v for v in names]
    flat_codes = {i for i, v in enumerate(norm) if v in FLAT_FAMILY_NAMES}
    return np.isin(code, np.array(sorted(flat_codes), dtype=code.dtype))


def select_ballast_rows(path: str, n_chunks: int, flat_frac: float = 0.0,
                        seed: int = 0) -> np.ndarray:
    """Row indices into ``path`` for the ballast, honouring the declared flat share.

    Args:
        path: The pre-training window ``.h5``.
        n_chunks: How many windows to carry. ``0`` disables the ballast.
        flat_frac: Share drawn from FLAT windows; the rest from terrain ones. ``0`` draws
            uniformly and never reads the family table.
        seed: Draw seed.

    Returns:
        Sorted row indices, exactly ``n_chunks`` of them.

    Raises:
        SystemExit: If the file is unreadable, if ``n_chunks`` exceeds what it holds, or if a
            stratum cannot fill its declared share.
    """
    if n_chunks <= 0:
        return np.zeros(0, dtype=np.int64)
    if not (0.0 <= flat_frac <= 1.0):
        raise SystemExit(f"ballast_flat_frac must be in [0, 1], got {flat_frac}")
    try:
        with h5py.File(path, "r") as f:
            n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
    except OSError as e:
        raise SystemExit(f"cannot read the ballast recording {path}: {e}") from None
    if n_chunks > n:
        raise SystemExit(
            f"asked for {n_chunks} ballast windows but {path} holds {n}. The ballast is a "
            "fixed slice of the pre-training set, so a short one is a different experiment "
            "from the declared one.")

    rng = np.random.default_rng(seed)
    if flat_frac <= 0.0:
        return np.sort(rng.choice(n, size=n_chunks, replace=False)).astype(np.int64)

    is_flat = flat_mask(path)
    want_flat = int(round(n_chunks * float(flat_frac)))
    want_terr = n_chunks - want_flat
    picked = []
    for name, want, pool in (("flat", want_flat, np.flatnonzero(is_flat)),
                             ("terrain", want_terr, np.flatnonzero(~is_flat))):
        if want == 0:
            continue
        if pool.size < want:
            raise SystemExit(
                f"the ballast asks for {want} {name} window(s) but {path} holds only "
                f"{pool.size}. Returning fewer would make the declared "
                f"ballast_flat_frac={flat_frac} a fiction while every log line read normally.")
        picked.append(rng.choice(pool, size=want, replace=False))
    return np.sort(np.concatenate(picked)).astype(np.int64)


def describe(path: str, rows: np.ndarray) -> str:
    """A one-line summary of what the ballast slice contains, for the run banner.

    Args:
        path: The window file.
        rows: The selected rows.

    Returns:
        The summary.
    """
    if rows.size == 0:
        return "ballast: none"
    try:
        flat = int(flat_mask(path)[rows].sum())
        share = f", {flat} flat / {rows.size - flat} terrain"
    except SystemExit:
        share = ""
    return f"ballast: {rows.size} window(s) from {path}{share}, weight 1, ungraded"
