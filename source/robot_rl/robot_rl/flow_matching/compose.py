# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pick which recorded windows go into a training set, given a per-bucket quota.

A recording is a rollout, so its window distribution is whatever the rollout happened to
produce: how long each clip is, how often its terrain came up, how far into an episode the
robot survived.  Training on it as-is means training on that accident.  This module turns a
recording into a DECLARED composition -- so many windows of flat, of stairs, of box -- by
selecting row indices, and it spreads each quota along the two axes that would otherwise
skew:

* **across clips** -- :func:`allocate_even` water-fills the quota over the clips in the
  bucket, so 40k stair windows come from as many distinct stair clips as the bucket has
  rather than from the handful with the most windows.  A clip with fewer windows than its
  fair share contributes all of them and its shortfall is redistributed.
* **along each clip** -- :func:`pick_evenly_spaced` takes its share at even spacing in
  ``clip_step``, so a clip's approach, its terrain crossing and its run-out are all
  represented rather than whichever phase the rollout dwelt on.

Selection is deterministic: no RNG, so the same recording and the same quotas give the same
dataset, and two compositions that share a bucket share those exact windows.

Nothing here touches HDF5 or torch -- it is index arithmetic over the recording's small
per-window index columns (``clip_id`` / ``clip_step`` / ``clip_family``), which is what
makes composing a 20 GB recording cheap enough to iterate on.
"""

from __future__ import annotations

import numpy as np


def allocate_even(available: np.ndarray, quota: int) -> np.ndarray:
    """Spread ``quota`` picks over items as evenly as their supply allows.

    Water-filling: every item with room takes an equal share of what is left, an item
    that runs out contributes all it has, and its shortfall is redistributed over the
    rest.  So a quota is met from the widest possible set of items, and the result is
    ``available`` itself when the quota exceeds the total supply.

    The final remainder (fewer picks left than items with room) goes to the items with the
    MOST room, ties broken by index.  That is at most one extra window per item, and
    resolving it deterministically is what keeps a composition reproducible.

    Args:
        available: ``[n]`` non-negative counts -- how many picks each item can supply.
        quota: Total picks wanted. Clamped to ``available.sum()``.

    Returns:
        ``[n]`` int64 picks per item, summing to ``min(quota, available.sum())``.

    Raises:
        ValueError: If ``quota`` is negative or any count is negative.
    """
    avail = np.asarray(available, dtype=np.int64)
    if quota < 0:
        raise ValueError(f"quota must be >= 0; got {quota}.")
    if avail.size and avail.min() < 0:
        raise ValueError("available counts must be >= 0.")
    take = np.zeros_like(avail)
    remaining = int(min(int(quota), int(avail.sum())))
    while remaining > 0:
        room = avail - take
        active = np.nonzero(room > 0)[0]
        share = remaining // active.size
        if share == 0:
            # Fewer picks left than items with room: one each to the roomiest, so the
            # remainder lands where it least distorts the per-item balance.
            order = active[np.lexsort((active, -room[active]))][:remaining]
            take[order] += 1
            break
        add = np.minimum(room[active], share)
        take[active] += add
        remaining -= int(add.sum())
    return take


#: Fractional bin offsets for :func:`select_windows`, one per clip: the golden-ratio
#: low-discrepancy sequence. A clip asked for ``k`` windows takes them one per bin of
#: ``n/k``, and WHERE in the bin is this offset -- fixed at 0.5 for every clip, a quota
#: below the bucket's clip count would take exactly one window from each clip and every
#: one of them would be that clip's midpoint, so a dataset even across clips would sit at
#: phase 0.5. Staggering the offset spreads those single picks across the phase range
#: while each clip's own picks stay evenly spaced. Deterministic, so a composition is
#: still reproducible.
_GOLDEN_RATIO_CONJUGATE: float = 0.6180339887498949


def bin_offsets(n_items: int) -> np.ndarray:
    """Per-item fractional bin offsets in ``[0, 1)`` (see :data:`_GOLDEN_RATIO_CONJUGATE`).

    Args:
        n_items: How many offsets to generate (one per clip in a bucket).

    Returns:
        ``[n_items]`` float64 offsets, low-discrepancy over ``[0, 1)``.
    """
    return ((np.arange(n_items, dtype=np.float64) + 1.0) * _GOLDEN_RATIO_CONJUGATE) % 1.0


def pick_evenly_spaced(n_available: int, k: int, offset: float = 0.5) -> np.ndarray:
    """Positions of ``k`` items spread evenly through ``n_available`` ordered ones.

    Bin sampling (``floor((i + offset) * n / k)``) rather than :func:`numpy.linspace`'s
    endpoints-included spacing: the endpoints of a clip's window list are its first and
    last usable step, and forcing both into every small sample over-weights clip starts
    and ends -- exactly the bias the composition exists to remove.

    Args:
        n_available: How many items there are.
        k: How many to take. Values ``>= n_available`` take everything.
        offset: Where in each bin of ``n/k`` to sample, in ``[0, 1)``. 0.5 is the bin
            midpoint; :func:`select_windows` varies it per clip (see :func:`bin_offsets`).

    Returns:
        Strictly increasing int64 positions into ``[0, n_available)``; length
        ``min(k, n_available)``.

    Raises:
        ValueError: If ``k`` is negative or ``offset`` is outside ``[0, 1)``.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0; got {k}.")
    if not (0.0 <= offset < 1.0):
        raise ValueError(f"offset must be in [0, 1); got {offset}.")
    if k >= n_available:
        return np.arange(n_available, dtype=np.int64)
    return np.floor((np.arange(k) + offset) * n_available / k).astype(np.int64)


def bucket_supply(bucket: np.ndarray, clip_id: np.ndarray) -> dict[str, dict[str, int]]:
    """What each bucket could supply: its window count and how many clips they span.

    Args:
        bucket: ``[W]`` per-window bucket name.
        clip_id: ``[W]`` per-window clip index.

    Returns:
        ``{bucket: {"windows", "clips"}}``, ordered by descending window count.
    """
    out: dict[str, dict[str, int]] = {}
    for name in np.unique(bucket):
        rows = bucket == name
        out[str(name)] = {
            "windows": int(rows.sum()),
            "clips": int(np.unique(clip_id[rows]).size),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["windows"]))


def select_windows(
    clip_id: np.ndarray,
    clip_step: np.ndarray,
    bucket: np.ndarray,
    quotas: dict[str, int],
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Choose the rows that make up a composed dataset.

    Each named bucket contributes its quota, spread evenly over the bucket's clips
    (:func:`allocate_even`) and evenly along each clip's timeline
    (:func:`pick_evenly_spaced`).  Buckets with no quota contribute nothing -- a
    composition states its whole content, so a bucket left out is left out on purpose
    rather than by default.

    Args:
        clip_id: ``[W]`` per-window clip index (the recording's ``clip_id``).
        clip_step: ``[W]`` per-window clip step of node 0 (the recording's ``clip_step``).
        bucket: ``[W]`` per-window bucket name (see :func:`window_buckets`).
        quotas: ``{bucket name: windows wanted}``.

    Returns:
        ``(rows, report)``: strictly increasing int64 row indices into the recording, and
        per-bucket ``{"requested", "taken", "clips_available", "clips_used"}``.  ``taken``
        is below ``requested`` exactly when the bucket ran out of windows.

    Raises:
        ValueError: If the three arrays disagree in length, a quota is negative, or a
            quota names a bucket the recording has no windows for.
    """
    clip_id = np.asarray(clip_id).reshape(-1)
    clip_step = np.asarray(clip_step).reshape(-1)
    bucket = np.asarray(bucket).reshape(-1)
    if not (clip_id.size == clip_step.size == bucket.size):
        raise ValueError(
            f"clip_id ({clip_id.size}), clip_step ({clip_step.size}) and bucket "
            f"({bucket.size}) must be the same length."
        )
    known = set(np.unique(bucket).tolist())
    missing = sorted(str(b) for b in quotas if str(b) not in {str(k) for k in known})
    if missing:
        raise ValueError(
            f"no windows for bucket(s) {missing}; this recording has "
            f"{sorted(str(k) for k in known)}."
        )

    picked: list[np.ndarray] = []
    report: dict[str, dict[str, int]] = {}
    for name, quota in quotas.items():
        if int(quota) < 0:
            raise ValueError(f"quota for {name!r} must be >= 0; got {quota}.")
        rows = np.nonzero(bucket == np.asarray(name, dtype=bucket.dtype))[0]
        # Clips of this bucket, and each clip's rows sorted by clip step -- "evenly along
        # the clip" is a statement about clip TIME, and the recording's row order is the
        # order envs happened to emit, which interleaves clips arbitrarily.
        clips, inverse = np.unique(clip_id[rows], return_inverse=True)
        per_clip = [rows[inverse == i][np.argsort(clip_step[rows[inverse == i]], kind="stable")]
                    for i in range(clips.size)]
        take = allocate_even(np.array([r.size for r in per_clip], dtype=np.int64), int(quota))
        offsets = bin_offsets(clips.size)
        chosen = [r[pick_evenly_spaced(r.size, int(t), float(u))]
                  for r, t, u in zip(per_clip, take, offsets) if t > 0]
        got = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
        picked.append(got)
        report[str(name)] = {
            "requested": int(quota),
            "taken": int(got.size),
            "clips_available": int(clips.size),
            "clips_used": int((take > 0).sum()),
        }

    rows = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
    # Sorted because every consumer wants it: HDF5 fancy indexing requires strictly
    # increasing indices, and it keeps the composed file in the source's own order.
    return np.unique(rows.astype(np.int64)), report


def window_buckets(
    clip_family: np.ndarray,
    family_names: list[str],
) -> np.ndarray:
    """Per-window bucket name at FAMILY granularity, from the recorded family code.

    The fine granularity: the eight terrain families the v5 library is built from
    (``family_stairs_up``, ``family_jump_off``, ...) plus ``unknown`` for clips whose tile
    has no manifest entry -- which for a terrain library means the flat clips, but is not
    renamed to "flat" here because the code genuinely says "no entry" and a library where
    that is not the flat clips would then be silently mislabelled.  Use
    :func:`window_buckets_by_type` when "flat" is what you mean.

    Args:
        clip_family: ``[W]`` per-window family code (the recording's ``clip_family``).
        family_names: Code -> name table (``RecordingMeta.family_names``).

    Returns:
        ``[W]`` array of bucket names.

    Raises:
        ValueError: If a code has no name (the recording and its meta disagree).
    """
    codes = np.asarray(clip_family).reshape(-1).astype(np.int64)
    names = list(family_names)
    if codes.size and (codes.max() >= len(names)):
        raise ValueError(
            f"clip_family holds code {int(codes.max())} but family_names has only "
            f"{len(names)} entries -- the recording and its metadata disagree."
        )
    return np.asarray(names, dtype=object)[codes]


#: The three ways a window can sit against terrain, from ``anchor_mask_is_pinned`` over the
#: nodes the generator predicts. A PARTITION of every window, and orthogonal to which clip
#: the window came from -- which is the whole reason it exists: a "transition" clip is a
#: flat approach, then the crossing, then the stair run, so its windows are all three of
#: these depending on ``clip_step``, and a quota on the clip type cannot ask for the event.
TERRAIN_SPAN_CLASSES: tuple[str, ...] = ("flat", "crossing", "on_terrain")


def window_buckets_by_terrain_span(
    clip_id: np.ndarray,
    clip_step: np.ndarray,
    pinned: np.ndarray,
    span_len: int,
    clip_len: np.ndarray | None = None,
    block: int = 65536,
) -> np.ndarray:
    """Per-window bucket name by what the window's own span does against terrain.

    Classifies the ``span_len`` nodes the window predicts, using the library's authored
    per-frame anchor mask (``motion_gen.world_lift.anchor_mask_is_pinned``: a frame is
    PINNED when the reference's placement comes from the terrain cell rather than from
    re-anchoring onto the robot, which is exactly the set of frames where the reference is
    interacting with terrain geometry):

    * ``flat`` -- no pinned frame in the span. Flat-ground walking, wherever it came from:
      a flat clip, a transition clip's approach, a box clip's run-out.
    * ``crossing`` -- the span holds both, so a registration BOUNDARY falls inside it. The
      transition event: the window is where the generator has to commit to geometry it is
      about to meet.
    * ``on_terrain`` -- every frame pinned. The stair climb or the box mount itself.

    Ground truth from the mask rather than a height threshold, for the reason recorded on
    the ``clip_step`` column: a threshold has to guess, and this does not.

    Args:
        clip_id: ``[W]`` per-window clip index.
        clip_step: ``[W]`` per-window clip step of node 0.
        pinned: ``[C, T]`` bool, the per-clip-per-step pinned flag (from
            ``ClipLibraryData.anchor_mask_per_step`` through ``anchor_mask_is_pinned``).
        span_len: Nodes per window (``F``, the recording's ``ref_future`` length).
        clip_len: ``[C]`` real clip lengths, to clamp a span that would run into the
            repeat-last padding. ``None`` uses the padded width.
        block: Windows classified per pass, bounding the gather's peak memory.

    Returns:
        ``[W]`` array of names from :data:`TERRAIN_SPAN_CLASSES`.

    Raises:
        ValueError: If ``span_len`` is not positive or a clip id falls outside ``pinned``.
    """
    cid = np.asarray(clip_id).reshape(-1).astype(np.int64)
    step = np.asarray(clip_step).reshape(-1).astype(np.int64)
    pinned = np.asarray(pinned, dtype=bool)
    if int(span_len) < 1:
        raise ValueError(f"span_len must be >= 1; got {span_len}.")
    if cid.size and (cid.max() >= pinned.shape[0] or cid.min() < 0):
        raise ValueError(
            f"clip_id spans [{int(cid.min())}, {int(cid.max())}] but the anchor mask covers "
            f"{pinned.shape[0]} clips -- the recording was made against a different library."
        )
    last = (np.full(pinned.shape[0], pinned.shape[1] - 1, dtype=np.int64) if clip_len is None
            else np.asarray(clip_len, dtype=np.int64) - 1)
    offs = np.arange(int(span_len), dtype=np.int64)
    out = np.empty(cid.size, dtype=object)
    for start in range(0, cid.size, int(block)):
        sl = slice(start, start + int(block))
        c, s = cid[sl], step[sl]
        # Clamp into the clip's REAL frames: past clip_len the library repeats the last
        # frame, whose mask would otherwise extend the final segment for free.
        idx = np.minimum(s[:, None] + offs[None, :], last[c][:, None])
        span = pinned[c[:, None], idx]
        any_p, all_p = span.any(1), span.all(1)
        out[sl] = np.where(all_p, "on_terrain", np.where(any_p, "crossing", "flat"))
    return out


def join_buckets(*axes: np.ndarray) -> np.ndarray:
    """Combine per-window bucket arrays into one compound key, joined by ``/``.

    So a composition can state a mix along more than one axis at once -- ``--by
    type,terrain`` gives buckets like ``transition/crossing``, which is the only way to ask
    for "the crossing events, and specifically the ones off a stair transition".

    Args:
        *axes: ``[W]`` bucket-name arrays, all the same length.

    Returns:
        ``[W]`` array of ``"a/b/..."`` names, components in the order given.

    Raises:
        ValueError: If no axes are given or they disagree in length.
    """
    if not axes:
        raise ValueError("join_buckets needs at least one axis.")
    sizes = {a.size for a in axes}
    if len(sizes) != 1:
        raise ValueError(f"axes disagree in length: {sorted(sizes)}")
    if len(axes) == 1:
        return np.asarray(axes[0], dtype=object)
    joined = np.asarray(axes[0], dtype=object).astype(str)
    for a in axes[1:]:
        joined = np.char.add(np.char.add(joined, "/"), np.asarray(a, dtype=object).astype(str))
    return joined.astype(object)


def window_buckets_by_type(
    clip_id: np.ndarray,
    clip_type_code: np.ndarray,
    type_names: list[str],
) -> np.ndarray:
    """Per-window bucket name at TERRAIN-TYPE granularity, joined from the clip library.

    The coarse granularity -- ``flat`` / ``terrain`` (a pure stair run) / ``transition``
    (flat onto stairs and back) / ``box`` -- which is what the library itself records per
    clip in ``terrain_type_code``.  Joined through ``clip_id`` rather than derived from the
    recorded family code, because the family -> type mapping lives in the library's terrain
    manifest and restating it here is exactly the kind of table that rots.

    Args:
        clip_id: ``[W]`` per-window clip index.
        clip_type_code: ``[C]`` per-CLIP terrain type code, indexed by clip id (the
            library's ``terrain_type_code``).
        type_names: Code -> name table (the library's ``terrain_type_names`` attr).

    Returns:
        ``[W]`` array of bucket names.

    Raises:
        ValueError: If a clip id falls outside the library, or a code has no name.
    """
    cid = np.asarray(clip_id).reshape(-1).astype(np.int64)
    codes = np.asarray(clip_type_code).reshape(-1).astype(np.int64)
    if cid.size and (cid.max() >= codes.size or cid.min() < 0):
        raise ValueError(
            f"clip_id spans [{int(cid.min())}, {int(cid.max())}] but the library has "
            f"{codes.size} clips -- the recording was made against a different library."
        )
    per_window = codes[cid]
    if per_window.size and per_window.max() >= len(type_names):
        raise ValueError(
            f"terrain_type_code holds {int(per_window.max())} but terrain_type_names has "
            f"only {len(type_names)} entries."
        )
    return np.asarray(list(type_names), dtype=object)[per_window]
