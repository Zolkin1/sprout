# Copyright 2026 Zachary Olkin. All rights reserved.

"""Merge several windowed FM recordings into ONE ``.h5``, streaming.

``flow_matching/train.py --data a.h5,b.h5`` concatenates recordings in memory: it loads
every file whole and then ``np.concatenate``s each array, so peak memory is ~2x the
combined uncompressed size.  That path works -- it is the one most runs here have used --
but the spike is real and grows with the file count.  Seven clutter variants at 1.07M
windows are ~78 GB resident, so the in-memory join would transiently need ~156 GB.

This script does the join once, on disk, holding ONE BLOCK of windows at a time instead
of two whole datasets, so the merge itself is flat regardless of input size.  ``train.py
--data merged.h5`` then takes the single-file path, which still has to hold the result --
what is avoided is the DOUBLING, not the residency.

(The original motivation was harsher: before ``train.py`` learned to skip the unused
exteroception array, a 22 GB pair was observed to wedge outright -- 37 minutes at 100%
CPU with zero user time and no allocation progress, cleared only by SIGTERM.  Selective
loading made that specific failure go away; the memory argument above is what remains.)

The semantics deliberately mirror ``train.py``'s in-memory version, so a merged file
trains identically to the comma-separated form:

* the FIRST file's :class:`RecordingMeta` wins (stored as the root JSON attrs);
* every input must carry the SAME set of window arrays, else the merge is refused;
* ``episode_id`` is offset per file past the running maximum, because the train/val split
  is BY EPISODE -- two files independently numbering from 0 would put one id in both
  halves and leak the split.

The copy is rank-agnostic (``shape = (total,) + src.shape[1:]``), so it carries the 4-D
camera-latency arrays -- ``depth_lag`` ``(N, n_lag, n_cam, px)`` and ``depth_lag_state``
-- as readily as a flat one.  Which arrays travel is ``recorded_data_io._WINDOW_DTYPES``,
the same table the recorder writes from, so a new array is picked up by both at once
rather than being silently dropped here.

Verified end to end on the seven-file clutter set (1,071,014 windows): counts, every
array's shape/dtype, byte-equality of sampled rows per source segment, and disjoint
increasing episode-id ranges.

NOTE what merging does NOT fix: variants of the SAME rollout (e.g. one recording
re-rendered with different visual clutter) share ``clip_id``/``clip_step`` but get
distinct offset ``episode_id``s, so the by-episode val split will put one variant of a
trajectory in train and another in val.  That is a property of the data, not of the join.

Usage::

    python scripts/rsl_rl/merge_recordings.py --out merged.h5 original.h5 augmented.h5
    python scripts/flow_matching/train.py --data merged.h5 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from recorded_data_io import (  # noqa: E402  isort: skip
    _WINDOW_DTYPES,
    _window_chunk,
)

#: Windows copied per read/write. 4096 x the widest row (depth, 2x780 float32) is ~50 MB,
#: big enough to amortize HDF5 chunk overhead and small enough that memory stays flat.
COPY_BLOCK = 4096


def _effective_len(f) -> int:
    """Return the number of VALID windows in an open recording.

    ``RecordedWindowWriter`` pre-sizes datasets to an upper bound and records how many were
    actually filled, so a run that stopped early leaves trailing garbage.
    :func:`load_recorded_windows` slices to ``num_windows_written``; this must agree or the
    merge would copy that garbage in as real windows.
    """
    return int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))


def _window_keys(f) -> list[str]:
    """Return the window arrays present in an open recording, in canonical order."""
    return [k for k in _WINDOW_DTYPES if k in f]


def merge(paths: list[str], out_path: str, compression: str | None = "lzf") -> None:
    """Stream-merge windowed recordings into one file.

    Args:
        paths: Input ``.h5`` recordings, in the order their windows should appear. The
            first one's metadata is carried over to the output.
        out_path: Output ``.h5`` path. Must not already exist.
        compression: h5py filter for the output datasets ("lzf" fast, "gzip" smaller,
            None off). Defaults to the writer's own "lzf".

    Raises:
        SystemExit: If an input is missing, holds no windows, the inputs disagree on which
            window arrays they carry, or the output already exists.
    """
    import h5py

    if os.path.exists(out_path):
        raise SystemExit(f"--out already exists, refusing to overwrite: {out_path}")
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"input not found: {p}")

    handles = [h5py.File(p, "r") for p in paths]
    try:
        counts = [_effective_len(f) for f in handles]
        for p, n in zip(paths, counts):
            if n <= 0:
                raise SystemExit(f"input holds no windows (num_windows_written=0): {p}")

        keys = _window_keys(handles[0])
        for p, f in zip(paths[1:], handles[1:]):
            other = _window_keys(f)
            if set(other) != set(keys):
                raise SystemExit(
                    "recordings carry different arrays and cannot be merged:\n"
                    f"  {sorted(set(keys) - set(other))} missing from {p}\n"
                    f"  {sorted(set(other) - set(keys))} missing from {paths[0]}"
                )
        if "episode_id" not in keys:
            raise SystemExit("inputs carry no 'episode_id'; cannot offset episodes safely.")

        total = sum(counts)
        print(f"[merge] {len(paths)} recordings: {counts} -> {total} windows")
        print(f"[merge] arrays: {keys}")

        with h5py.File(out_path, "w") as out:
            # Datasets sized to the exact total, shaped/typed from the first input so the
            # output is indistinguishable from a single long recording.
            for key in keys:
                src = handles[0][key]
                shape = (total,) + src.shape[1:]
                out.create_dataset(
                    key, shape=shape, dtype=src.dtype,
                    chunks=_window_chunk(shape), compression=compression,
                )

            # First file's meta wins, matching train.py. The count attrs are ours to set.
            for k, v in handles[0].attrs.items():
                if k not in ("num_windows", "num_windows_written"):
                    out.attrs[k] = v

            dst = 0
            ep_off = 0
            for p, f, n in zip(paths, handles, counts):
                ep_max = -1
                for key in keys:
                    src = f[key]
                    for i in range(0, n, COPY_BLOCK):
                        j = min(i + COPY_BLOCK, n)
                        block = src[i:j]
                        if key == "episode_id":
                            block = np.asarray(block) + ep_off
                            ep_max = max(ep_max, int(block.max()))
                        out[key][dst + i:dst + j] = block
                print(f"[merge]   {os.path.basename(p)}: {n} windows at offset {dst}, "
                      f"episode ids +{ep_off} (max now {ep_max})")
                dst += n
                # Next file starts past every id used so far, so the val split cannot leak.
                ep_off = ep_max + 1

            out.attrs["num_windows"] = total
            out.attrs["num_windows_written"] = total
    finally:
        for f in handles:
            f.close()

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"[merge] wrote {out_path} ({size_gb:.1f} GB)")


def verify(out_path: str, expected: int) -> None:
    """Reload the merged file the way training will and check what came back.

    Args:
        out_path: The merged ``.h5``.
        expected: Window count the merge reported writing.

    Raises:
        SystemExit: If the reload disagrees on the count or the episode ids are not unique
            per source block (which would mean the offsetting failed and the split leaks).
    """
    from recorded_data_io import load_recorded_windows  # noqa: PLC0415

    data, meta = load_recorded_windows(out_path)
    n = int(np.asarray(data["episode_id"]).shape[0])
    if n != expected:
        raise SystemExit(f"verify FAILED: reload gave {n} windows, merge wrote {expected}")
    eps = np.asarray(data["episode_id"])
    print(f"[verify] {n} windows, {len(np.unique(eps))} unique episode ids "
          f"(range {int(eps.min())}..{int(eps.max())})")
    print(f"[verify] arrays: {sorted(data)}")
    print(f"[verify] meta: dt={meta.dt}, task={meta.task_name}, "
          f"history_len={getattr(meta, 'history_len', '?')}")
    print("[verify] OK")


def main() -> None:
    """Parse arguments and run the merge."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Windowed .h5 recordings to merge, in order.")
    ap.add_argument("--out", required=True, help="Output .h5 path (must not exist).")
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"],
                    help="Output compression filter. 'lzf' matches the recorder.")
    ap.add_argument("--no_verify", action="store_true",
                    help="Skip the post-merge reload check (it re-reads the whole file).")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("give at least two recordings to merge.")

    comp = None if args.compression == "none" else args.compression
    merge(args.inputs, args.out, compression=comp)
    if not args.no_verify:
        with __import__("h5py").File(args.out, "r") as f:
            total = _effective_len(f)
        verify(args.out, total)


if __name__ == "__main__":
    main()
