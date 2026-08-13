# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compose an FM training set out of a recording, with a DECLARED mix.

A recording is a rollout, so what it contains is an accident of the rollout: how many tiles
of each terrain the scene has, how long each clip is, how far the robot got.  On
``g1_mb_v5_body`` that lands at 31% flat / 34% box / 20% transition / 15% terrain by clip,
and -- the number that usually matters more -- only **9%** of windows have a terrain
crossing inside them at all.  This script replaces that accident with a statement, and
spreads each quota evenly over the bucket's clips and evenly along each clip's timeline
(:mod:`robot_rl.flow_matching.compose`).

**What a bucket is, is your choice of axis** (``--by``), because the useful questions live
at two different levels:

* ``type`` -- the CLIP's terrain: ``flat`` / ``terrain`` (a pure stair run) / ``transition``
  (flat onto stairs and back) / ``box``, joined from the library's ``terrain_type_code``.
* ``family`` -- the CLIP's family: ``family_stairs_up``, ``family_jump_off``, ... from the
  recording's own per-window tag.
* ``terrain`` -- what the WINDOW ITSELF spans: ``flat`` / ``crossing`` / ``on_terrain``,
  from the library's authored per-frame anchor mask.  This is the one a clip-level axis
  cannot express: a transition clip is a flat approach, then the crossing, then the stair
  run, so ``--by type --mix transition=40000`` buys mostly ordinary flat walking -- 47% of
  a transition clip's windows are pure flat and only 40% contain the event.

Axes compose: ``--by type,terrain`` keys on ``transition/crossing``, ``box/on_terrain``, and
so on, which is how you ask for the crossings of one terrain in particular.

The intended workflow is to record ONCE, broadly (``record_data.py`` defaults ``--start_step
0 --clip_draw cycle``, which spans the library and covers each clip's phase uniformly), then
compose as many training sets from it as there are mixes worth trying.  Composing decides
from the recording's small index columns, so choosing a mix is instant; the copy that
follows is the only expensive part.

Examples:
    # What is in there, along every axis?
    python scripts/flow_matching/compose_dataset.py rec.h5 --list

    # A third of the data is the terrain event itself.
    python scripts/flow_matching/compose_dataset.py rec.h5 -o mix.h5 --by terrain \
        --mix crossing=40000 --mix on_terrain=40000 --mix flat=40000

    # The same, but splitting the crossings between stairs and box.
    python scripts/flow_matching/compose_dataset.py rec.h5 -o mix.h5 --by type,terrain \
        --mix transition/crossing=20000 --mix box/crossing=20000 \
        --mix terrain/on_terrain=30000 --mix flat/flat=40000
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from robot_rl.flow_matching.compose import (
    bucket_supply,
    join_buckets,
    select_windows,
    window_buckets,
    window_buckets_by_terrain_span,
    window_buckets_by_type,
)

#: Index columns read to DECIDE the composition. Kilobytes even for a million windows,
#: which is what makes trying a mix cheap; the heavy arrays are only ever streamed.
_INDEX_KEYS = ("clip_id", "clip_step", "clip_family")

#: Axes a bucket key can be built from (see the module docstring).
_AXES = ("type", "family", "terrain")

#: Windows per block in the streaming copy. Large enough to amortise HDF5 chunk reads
#: (chunks are a few hundred windows), small enough that one block of the widest array
#: (a dense height scan at ~29 KiB/window) stays well under a gigabyte.
_COPY_BLOCK: int = 8192


def _parse_quota(spec: str) -> tuple[str, int]:
    """Parse one ``NAME=COUNT`` quota.

    Args:
        spec: The raw ``--mix`` value.

    Returns:
        ``(name, count)``.

    Raises:
        argparse.ArgumentTypeError: If the spec is not ``NAME=<non-negative int>``.
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected NAME=COUNT, got {spec!r}")
    name, _, count = spec.partition("=")
    name = name.strip()
    try:
        n = int(count)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{spec!r}: {count!r} is not an integer") from None
    if not name:
        raise argparse.ArgumentTypeError(f"{spec!r}: empty bucket name")
    if n < 0:
        raise argparse.ArgumentTypeError(f"{spec!r}: count must be >= 0")
    return name, n


def _load_index(path: str) -> tuple[dict[str, np.ndarray], dict, int, int]:
    """Read a recording's per-window index columns, its metadata attrs, and its span.

    Args:
        path: The recording ``.h5``.

    Returns:
        ``(index arrays, decoded attrs, num_windows_written, future_len)``.

    Raises:
        KeyError: If the recording predates the ``clip_id`` / ``clip_step`` columns, which
            a composition cannot be built without.
    """
    with h5py.File(path, "r") as f:
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        missing = [k for k in ("clip_id", "clip_step") if k not in f]
        if missing:
            raise KeyError(
                f"{path} has no {missing} column(s), so its windows cannot be traced back to "
                "clips. Re-record with the current record_data.py."
            )
        index = {k: f[k][:n] for k in _INDEX_KEYS if k in f}
        future_len = int(f["ref_future"].shape[1])
        attrs = {k: json.loads(v) for k, v in f.attrs.items()
                 if k not in ("num_windows", "num_windows_written")}
    return index, attrs, n, future_len


def _load_library(clip_path: str):
    """Load ONLY a clip library's per-clip metadata (no dense arrays -- under a second).

    Args:
        clip_path: The clip library ``.h5`` the recording was made against.

    Returns:
        The :class:`motion_gen.clips.clip_io.ClipLibraryData`, dense fields empty.

    Raises:
        FileNotFoundError: If the library is not on disk.
    """
    from motion_gen.clips.clip_io import load_clip_library  # noqa: PLC0415

    if not os.path.isfile(clip_path):
        raise FileNotFoundError(
            f"clip library {clip_path!r} not found. The 'type' and 'terrain' axes are joined "
            "from the library the recording was made against; pass --clip, or use --by family."
        )
    return load_clip_library(clip_path, device="cpu", keys=[])


def _build_axis(axis: str, index: dict, attrs: dict, lib, future_len: int) -> np.ndarray:
    """Per-window bucket names for one axis.

    Args:
        axis: One of :data:`_AXES`.
        index: The recording's index columns.
        attrs: The recording's decoded metadata attrs.
        lib: The clip library (``None`` when only the recording is needed).
        future_len: ``F``, the window's span in clip steps.

    Returns:
        ``[W]`` bucket names.

    Raises:
        ValueError: If the axis needs something this recording / library cannot supply.
    """
    from motion_gen.world_lift import anchor_mask_is_pinned  # noqa: PLC0415

    if axis == "family":
        names = attrs.get("family_names") or []
        if "clip_family" not in index or not names:
            raise ValueError(
                "this recording carries no clip_family tag / family_names, so --by family "
                "cannot be resolved. Use --by type or --by terrain."
            )
        return window_buckets(index["clip_family"], names)
    if axis == "type":
        type_names = lib.attrs.get("terrain_type_names")
        if lib.terrain_type_code.numel() == 0 or not type_names:
            raise ValueError(
                "the clip library carries no terrain_type_code / terrain_type_names, so "
                "--by type cannot be resolved. Use --by family."
            )
        return window_buckets_by_type(
            index["clip_id"], lib.terrain_type_code.numpy(), type_names)
    if axis == "terrain":
        pinned = anchor_mask_is_pinned(lib.anchor_mask_per_step("cpu")).numpy()
        return window_buckets_by_terrain_span(
            index["clip_id"], index["clip_step"], pinned, future_len,
            clip_len=lib.clip_len_or_full("cpu").numpy(),
        )
    raise ValueError(f"unknown axis {axis!r}; expected one of {_AXES}.")


def _copy_selected(src_path: str, dst_path: str, rows: np.ndarray, attrs: dict) -> None:
    """Stream the selected windows into a new recording, mirroring the source's layout.

    Every dataset is recreated with the source's own dtype, chunking and compression and
    filled block by block, so the composed file is a faithful subset that
    ``load_recorded_windows`` reads exactly like the original -- no re-typing, and never
    more than one block of one array in RAM.

    Args:
        src_path: Source recording.
        dst_path: Output path (overwritten).
        rows: Strictly increasing row indices to keep.
        attrs: Root attrs to write (already JSON-decoded; re-encoded here).
    """
    n_out = int(rows.size)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        keys = [k for k in src if isinstance(src[k], h5py.Dataset)]
        for key in keys:
            d = src[key]
            dst.create_dataset(
                key, shape=(n_out,) + d.shape[1:], dtype=d.dtype,
                chunks=((min(n_out, d.chunks[0]),) + d.chunks[1:]) if d.chunks else None,
                compression=d.compression, compression_opts=d.compression_opts,
            )
        for k, v in attrs.items():
            dst.attrs[k] = json.dumps(v)
        dst.attrs["num_windows"] = n_out
        dst.attrs["num_windows_written"] = n_out
        for start in range(0, n_out, _COPY_BLOCK):
            block = rows[start:start + _COPY_BLOCK]
            for key in keys:
                dst[key][start:start + block.size] = src[key][block]
            print(f"\r[compose] copied {min(start + _COPY_BLOCK, n_out)}/{n_out} windows",
                  end="", flush=True)
    print()


def main() -> None:
    """Parse the requested mix, select the rows, and write the composed recording."""
    p = argparse.ArgumentParser(
        description="Compose an FM training set from a recording with a declared mix.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument("recording", type=str, help="Source windowed recording (.h5).")
    p.add_argument("-o", "--out", type=str, default=None,
                   help="Output .h5. Required unless --list.")
    p.add_argument("--by", type=str, default=None, metavar="AXIS[,AXIS]",
                   help=f"Which axis (or axes, joined by '/') a bucket key is: {', '.join(_AXES)}. "
                        "'type'/'family' describe the CLIP; 'terrain' describes what the WINDOW "
                        "itself spans (flat / crossing / on_terrain).")
    p.add_argument("--mix", type=_parse_quota, action="append", default=[], metavar="NAME=N",
                   dest="mix", help="Windows wanted from one bucket. Repeatable. Buckets with no "
                                    "quota contribute nothing.")
    p.add_argument("--clip", type=str, default=None,
                   help="Clip library the recording was made against, for the 'type' and "
                        "'terrain' axes. Default: the recording's own clip_library_path.")
    p.add_argument("--list", action="store_true", default=False,
                   help="Print what each bucket could supply and exit. With --by, for that key; "
                        "without, for every axis separately.")
    args = p.parse_args()

    index, attrs, n_windows, future_len = _load_index(args.recording)
    print(f"[compose] {args.recording}: {n_windows} windows (F={future_len}) over "
          f"{np.unique(index['clip_id']).size} clips")

    axes = [a.strip() for a in (args.by or "").split(",") if a.strip()]
    bad = [a for a in axes if a not in _AXES]
    if bad:
        p.error(f"unknown --by axis {bad}; expected from {list(_AXES)}.")

    clip_path = args.clip or attrs.get("clip_library_path", "")
    needs_lib = (not axes) or any(a in ("type", "terrain") for a in axes)
    lib = _load_library(clip_path) if needs_lib else None

    if args.list:
        for axis in (axes or _AXES):
            try:
                bucket = _build_axis(axis, index, attrs, lib, future_len)
            except (ValueError, KeyError) as exc:
                print(f"\n  --by {axis}: unavailable ({exc})")
                continue
            print(f"\n  --by {axis}:")
            for name, s in bucket_supply(bucket, index["clip_id"]).items():
                print(f"    {name:28s} {s['windows']:8d} windows over {s['clips']:5d} clips "
                      f"({100.0 * s['windows'] / max(n_windows, 1):5.1f}%)")
        if len(axes) > 1:
            joined = join_buckets(*[_build_axis(a, index, attrs, lib, future_len) for a in axes])
            print(f"\n  --by {','.join(axes)}:")
            for name, s in bucket_supply(joined, index["clip_id"]).items():
                print(f"    {name:28s} {s['windows']:8d} windows over {s['clips']:5d} clips "
                      f"({100.0 * s['windows'] / max(n_windows, 1):5.1f}%)")
        return

    if not axes:
        p.error("--by is required (or --list to see what is available).")
    if not args.mix:
        p.error("--mix NAME=COUNT is required; a composition states its whole content.")
    if not args.out:
        p.error("--out is required unless --list.")

    bucket = join_buckets(*[_build_axis(a, index, attrs, lib, future_len) for a in axes])
    quotas = dict(args.mix)
    rows, report = select_windows(index["clip_id"], index["clip_step"], bucket, quotas)
    print(f"[compose] selected {rows.size} of {n_windows} windows:")
    for name, r in report.items():
        short = "" if r["taken"] >= r["requested"] else "  <- SHORT"
        print(f"    {name:28s} {r['taken']:8d} / {r['requested']:8d} windows over "
              f"{r['clips_used']}/{r['clips_available']} clips{short}")
    if rows.size == 0:
        raise SystemExit("[compose] the requested mix selects no windows.")

    # Provenance: the composed file states what it was cut from and how, since its own
    # arrays no longer show it -- and its num_pairs must match its new length.
    attrs = dict(attrs)
    attrs["num_pairs"] = int(rows.size)
    attrs["content_sha256"] = ""  # no longer the source's content; not recomputed here
    attrs["composed_from"] = {
        "source": os.path.abspath(args.recording),
        "source_windows": n_windows,
        "by": axes,
        "quotas": quotas,
        "report": report,
        "clip_library_path": clip_path,
    }
    _copy_selected(args.recording, args.out, rows, attrs)
    print(f"[compose] wrote {rows.size} windows to {args.out} "
          f"({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
