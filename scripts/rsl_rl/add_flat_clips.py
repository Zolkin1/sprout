#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Add flat-ground clips (and therefore flat TILES) to a terrain-only clip library.

The small viewing libraries (``clips/g1_mb_v5_small*.h5``) contain stairs and boxes only, so
anything that needs flat ground -- notably the SONIC baseline's fidelity gate, which must
establish that a tracker works at all before terrain numbers mean anything -- has to load the
full 2.7 GB library. This copies a handful of flat clips out of a donor library so a small
library gains flat ground too.

Two things have to be right for the RL side to actually build flat tiles:

* **flat clips**, tagged ``terrain_id == -1``, which is how the reset recognises flat ground; and
* ``generator_config.counts.flat``, which is what
  :func:`...terrains.clip_terrain_loader.default_num_flat_tiles` reads to decide how many flat
  tiles to build. Flat tiles are NOT one-per-clip and carry no ``terrain_manifest`` entry -- every
  flat clip is sampled onto any flat tile -- so the clip count and the tile count are independent
  knobs.

The flat tile's footprint is sized from the widest pelvis travel over ALL flat clips
(``_flat_clip_travel``), so adding a wandering 20 s clip makes the flat tile large and the scene
awkward to look at. The chosen clips' span is reported, and ``--max_travel_m`` filters on it.

The source library's own clips, terrain tiles and manifest are copied through untouched; only the
time dimension is padded if the donor's clips are longer (repeat-last-frame, the format's existing
convention -- ``clip_len`` bounds the real data, so the padding is never read).

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/add_flat_clips.py \\
        --src clips/g1_mb_v5_small.h5 --donor clips/g1_mb_v5.h5 \\
        --flat_clips 6 --flat_tiles 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import h5py
import numpy as np

# Datasets with a leading clip axis and a time axis that needs padding.
_TIME_DATASETS = ("y_des", "dy_des", "contact", "skill_id", "stride_idx", "target_vel",
                  "anchor_mask_t")
# Datasets with a leading clip axis and no time axis.
_PER_CLIP_DATASETS = ("terrain_id", "terrain_type_code", "anchor_mask", "clip_len")

_LAYOUT_ATTRS = ("dt", "pos_output_names", "vel_output_names", "contact_frames", "ref_frame",
                 "terrain_type_names")


def _attr(f: h5py.File, key: str):
    """Read an attr, decoding the JSON the library format stores it as."""
    v = f.attrs.get(key)
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def _pad_time(arr: np.ndarray, t_out: int) -> np.ndarray:
    """Pad (or truncate) a ``[n, T, ...]`` array to ``t_out`` by repeating its last frame."""
    t = arr.shape[1]
    if t == t_out:
        return arr
    if t > t_out:
        return arr[:, :t_out]
    tail = np.repeat(arr[:, -1:], t_out - t, axis=1)
    return np.concatenate([arr, tail], axis=1)


def _pelvis_xy_indices(pos_names: list[str], ref_frame: str) -> tuple[int, int]:
    """Column indices of the reference frame's world x and y in ``y_des``."""
    return pos_names.index(f"{ref_frame}:pos_x"), pos_names.index(f"{ref_frame}:pos_y")


def _travel(y: np.ndarray, ix: int, iy: int, clip_len: np.ndarray) -> np.ndarray:
    """Per-clip pelvis xy bounding-box diagonal over the clip's REAL frames."""
    out = np.empty(len(y))
    for i in range(len(y)):
        n = max(int(clip_len[i]), 1)
        xy = y[i, :n][:, [ix, iy]]
        out[i] = float(np.hypot(*(xy.max(0) - xy.min(0))))
    return out


def main() -> int:
    """Copy flat clips from a donor library into a terrain-only one."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--src", required=True, help="Terrain-only library to extend.")
    p.add_argument("--donor", required=True, help="Library to take flat clips from.")
    p.add_argument("--out", default=None,
                   help="Output path. Default: overwrite --src, keeping a .pre_flat.h5 backup.")
    p.add_argument("--flat_clips", type=int, default=6, help="How many flat clips to copy.")
    p.add_argument("--flat_tiles", type=int, default=2,
                   help="Flat TILES the RL importer should build (generator_config.counts.flat).")
    p.add_argument("--flat_clip_ids", default=None,
                   help="Comma-separated donor clip indices to use instead of picking.")
    p.add_argument("--max_travel_m", type=float, default=12.0,
                   help="Skip flat clips whose pelvis wanders further than this, so the flat tile "
                        "(sized to the widest clip) stays small enough to look at.")
    args = p.parse_args()

    out = args.out or args.src
    with h5py.File(args.src, "r") as s, h5py.File(args.donor, "r") as d:
        for k in _LAYOUT_ATTRS:
            a, b = _attr(s, k), _attr(d, k)
            if a != b:
                raise SystemExit(f"[flat] layout attr '{k}' differs between libraries "
                                 f"({a!r} vs {b!r}); they are not splice-compatible.")
        pos_names = list(_attr(s, "pos_output_names"))
        ref_frame = _attr(s, "ref_frame")
        dt = float(_attr(s, "dt"))

        donor_tid = d["terrain_id"][:]
        flat_pool = np.nonzero(donor_tid == -1)[0]
        if flat_pool.size == 0:
            raise SystemExit(f"[flat] {args.donor} has no flat clips (terrain_id == -1).")

        if args.flat_clip_ids:
            chosen = np.asarray([int(x) for x in args.flat_clip_ids.split(",")], dtype=np.int64)
            bad = [int(c) for c in chosen if donor_tid[c] != -1]
            if bad:
                raise SystemExit(f"[flat] donor clips {bad} are not flat (terrain_id != -1).")
        else:
            # Walk the pool in order and keep the first clips that are compact enough, so the
            # selection is deterministic and traceable back to donor clip ids.
            ix, iy = _pelvis_xy_indices(pos_names, ref_frame)
            chosen, scanned = [], 0
            for cid in flat_pool:
                if len(chosen) >= args.flat_clips:
                    break
                scanned += 1
                n = int(d["clip_len"][cid])
                xy = d["y_des"][cid, :max(n, 1)][:, [ix, iy]]
                if float(np.hypot(*(xy.max(0) - xy.min(0)))) <= args.max_travel_m:
                    chosen.append(int(cid))
            chosen = np.asarray(chosen, dtype=np.int64)
            print(f"[flat] scanned {scanned} donor flat clips, kept {len(chosen)} within "
                  f"{args.max_travel_m} m travel")
        if chosen.size == 0:
            raise SystemExit("[flat] no flat clip passed --max_travel_m; raise it.")

        n_src = int(s["terrain_id"].shape[0])
        t_out = max(int(s["y_des"].shape[1]), int(d["y_des"].shape[1]))
        n_out = n_src + int(chosen.size)
        print(f"[flat] {args.src}: {n_src} clips x {s['y_des'].shape[1]} steps")
        print(f"[flat] adding donor clips {chosen.tolist()} -> {n_out} clips x {t_out} steps")

        gen_cfg = _attr(s, "generator_config") or {}
        gen_cfg.setdefault("counts", {})
        gen_cfg["counts"]["flat"] = int(args.flat_tiles)

        tmp = out + ".tmp"
        with h5py.File(tmp, "w") as o:
            for key in _TIME_DATASETS + _PER_CLIP_DATASETS:
                if key not in s:
                    continue
                src_arr = s[key][:]
                don_arr = d[key][chosen] if key in d else None
                if key in _TIME_DATASETS:
                    src_arr = _pad_time(src_arr, t_out)
                    don_arr = _pad_time(don_arr, t_out) if don_arr is not None else None
                if don_arr is None:
                    raise SystemExit(f"[flat] donor is missing dataset '{key}'.")
                merged = np.concatenate([src_arr, don_arr.astype(src_arr.dtype, copy=False)], 0)
                # clip_len must never exceed the (possibly truncated) time axis.
                if key == "clip_len":
                    merged = np.minimum(merged, t_out)
                o.create_dataset(key, data=merged, chunks=(1,) + merged.shape[1:],
                                 compression="lzf")
            for k in s.attrs:
                o.attrs[k] = s.attrs[k]
            o.attrs["num_steps"] = json.dumps(t_out)
            o.attrs["clip_length_s"] = json.dumps(round(t_out * dt, 6))
            o.attrs["num_clips"] = n_out
            o.attrs["num_clips_written"] = n_out
            o.attrs["generator_config"] = json.dumps(gen_cfg)

        # Report the flat tile the RL side will build from these clips.
        ix, iy = _pelvis_xy_indices(pos_names, ref_frame)
        with h5py.File(tmp, "r") as o:
            tid = o["terrain_id"][:]
            fl = np.nonzero(tid == -1)[0]
            trav = _travel(o["y_des"][fl], ix, iy, o["clip_len"][:][fl])
        print(f"[flat] flat clips now in the library: {len(fl)}; pelvis travel "
              f"min {trav.min():.2f} m / max {trav.max():.2f} m "
              f"(the flat TILE is sized to the max)")
        print(f"[flat] generator_config.counts.flat = {args.flat_tiles} -> "
              f"{args.flat_tiles} flat tile(s) will be built")

    if out == args.src:
        backup = os.path.splitext(args.src)[0] + ".pre_flat.h5"
        if not os.path.exists(backup):
            shutil.copy2(args.src, backup)
            print(f"[flat] backed up the original to {backup}")
    os.replace(tmp, out)
    print(f"[flat] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
