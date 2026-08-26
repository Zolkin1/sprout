#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compose several clip libraries into ONE RL training library at controlled tile proportions.

The RL importer distributes envs ROUND-ROBIN across all tiles, so a family's share of training =
(its tile count) / (total tiles). This tool merges source libraries and sets each family's tile
count to the requested target -- replicating a family's distinct tiles when the target exceeds what
the source has, subsetting when it's fewer -- so you dial the training distribution directly.

    --mix "clips/g1_box_v1.h5:10, clips/g1_distractor_v1.h5:30, clips/g1_flat_v1.h5:60"

Each ``--mix`` entry is ``path:tiles``. A terrain source contributes one manifest tile per (replicated)
terrain_id. A source whose clips are all terrain_id -1 is a FLAT pool (its ``:tiles`` -> flat-tile count).

Flat tiles can instead be drawn from a specific source with ``--flat-from`` -- e.g. a stairs+flat library
like ``g1_terrain_v2`` provides BOTH its stair tiles (via ``--mix``) and its large wandering flat clips
(via ``--flat-from``), so flat tiles are sized to the largest clip again. ``--flat-tiles`` sets the count.
``--max-clips-per-tile`` caps clips per terrain tile (small = lightweight dry-run library; 0 = full training).

Runs with NO IsaacLab.

    # full dry-run library: stairs + full boxes + pure boxes + distractors + large flat (lightweight)
    python scripts/rsl_rl/compose_terrain_library.py \
        --mix "clips/g1_terrain_v2.h5:12, clips/g1_box_v1.h5:8, clips/g1_box_pure_v1.h5:2, clips/g1_distractor_v1.h5:12" \
        --flat-from clips/g1_terrain_v2.h5 --flat-tiles 20 --flat-clips 200 \
        --max-clips-per-tile 6 --out clips/g1_train_full_v1.h5
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from motion_gen.clips.clip_fingerprint import fingerprint_clip_config, fingerprint_solved_library
from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library

_PERSTEP = ["y_des", "dy_des", "contact", "target_vel", "anchor_mask_t"]


def _pad(t: torch.Tensor, T: int) -> torch.Tensor:
    """Pad (repeat last frame) or truncate a ``[t0, ...]`` tensor to length ``T`` along dim 0."""
    n = t.shape[0]
    if n == T:
        return t
    if n > T:
        return t[:T]
    return torch.cat([t, t[-1:].expand((T - n,) + tuple(t.shape[1:]))], dim=0)


_LIB_CACHE: dict = {}


def _get_lib(path: str):
    """Load a clip library once and cache it (terrain/flat sources can be multi-GB)."""
    if path not in _LIB_CACHE:
        _LIB_CACHE[path] = load_clip_library(path, "cpu")
    return _LIB_CACHE[path]


def _sample(idxs: list[int], k: int) -> list[int]:
    """Evenly subsample ``idxs`` down to at most ``k`` (``k <= 0`` keeps all)."""
    if k <= 0 or len(idxs) <= k:
        return list(idxs)
    return list(idxs)[:: max(1, len(idxs) // k)][:k]


# Meta keys that MUST agree across sources: the composed library keeps ONE meta (the first source's), so a
# mismatch here silently reinterprets another source's data -- different dt, or y_des/dy_des/contact columns
# meaning different channels.  ``skill_list`` only labels the (RL-unused) skill_id dataset, so it warns.
_META_MUST_MATCH = ("dt", "pos_output_names", "vel_output_names", "contact_frames", "ref_frame")
_META_SHOULD_MATCH = ("skill_list",)


def _fmt(v) -> str:
    """Compact display for a meta value (long channel-name lists get elided)."""
    if isinstance(v, list) and len(v) > 4:
        return f"[{len(v)} items: {v[0]}, {v[1]}, ..., {v[-1]}]"
    return repr(v)


def _validate_meta(paths: list[str]) -> None:
    """Fail loudly if the sources' clip metas are not composable.

    The writer stores one meta for the whole library, so composing sources whose channel layout, control dt
    or reference frame differ produces a library that is silently wrong rather than obviously broken -- worth
    an explicit gate now that libraries come from more than one generator (DP matcher vs Motion Bricks).
    """
    ref_path, rest = paths[0], paths[1:]
    ref = _get_lib(ref_path).meta
    problems: list[str] = []
    soft: dict[str, set[str]] = {}
    for path in rest:
        m = _get_lib(path).meta
        for k in _META_MUST_MATCH:
            if m.get(k) != ref.get(k):
                problems.append(f"  {k}: {ref_path} has {_fmt(ref.get(k))}\n"
                                f"      {' ' * len(k)}{path} has {_fmt(m.get(k))}")
        for k in _META_SHOULD_MATCH:
            if m.get(k) != ref.get(k):
                soft.setdefault(k, set()).add(path.split("/")[-1])
    # One line per key, not per source: composing per-family libraries always differs in skill_list, and a
    # warning per source buries the real errors.
    for k, srcs in soft.items():
        print(f"[compose][WARN] {k} differs across {len(srcs)} source(s) ({', '.join(sorted(srcs))}); "
              f"keeping {ref_path.split('/')[-1]}'s {_fmt(ref.get(k))}. Only the (RL-unused) skill_id "
              f"dataset is labelled by it.")
    if problems:
        raise SystemExit("[compose] sources are not composable -- incompatible clip meta:\n"
                         + "\n".join(problems)
                         + "\n[compose] regenerate the odd source with the same --source / --dt, or drop it.")


def _load_source(path: str, types: set[str] | None = None):
    """-> (is_flat, tiles). ``tiles`` = list of (manifest_entry|None, [clip_idx]) grouped by
    terrain_id (one 'tile' each); a source of only terrain_id -1 clips is flat (one pseudo-tile).

    ``types`` keeps only tiles whose manifest ``type`` is listed -- so a single ALREADY-COMPOSED
    library (stairs + transitions + boxes in one file) can be narrowed to one family without
    regenerating anything.  That keeps a single-family experiment otherwise identical to the mixed
    run: same clips, same motion, only the terrain distribution changes."""
    lib = _get_lib(path)
    tid = lib.terrain_id
    if bool((tid == -1).all()):
        return True, [(None, list(range(lib.num_clips)))]
    man = {int(m["terrain_id"]): m for m in lib.attrs["terrain_manifest"]}
    tiles = [(man[t], (tid == t).nonzero().flatten().tolist())
             for t in sorted({int(v) for v in tid.tolist()} - {-1})]
    if types is not None:
        available = sorted({str(m.get("type")) for m, _ in tiles})
        tiles = [(m, idx) for m, idx in tiles if str(m.get("type")) in types]
        if not tiles:
            raise SystemExit(
                f"[compose] no tiles of type(s) {sorted(types)} in {path}; it has {available}"
            )
    return False, tiles


def _clip_batch(lib, idx: int, T: int, new_tid: int, dev):
    """Per-clip batch dict for source clip ``idx``, padded to ``T``, retagged to ``new_tid``."""
    has_amt = lib.anchor_mask_t.numel() > 0
    b = {
        "y_des": _pad(lib.y_des[idx], T).unsqueeze(0),
        "dy_des": _pad(lib.dy_des[idx], T).unsqueeze(0),
        "contact": _pad(lib.contact[idx], T).unsqueeze(0),
        "skill_id": _pad(lib.skill_id[idx], T).unsqueeze(0),
        "stride_idx": _pad(lib.stride_idx[idx], T).unsqueeze(0),
        "target_vel": _pad(lib.target_vel[idx], T).unsqueeze(0),
        "terrain_id": torch.tensor([new_tid], dtype=torch.int32),
        "terrain_type": lib.attrs.get("terrain_type_names", ["flat", "terrain", "transition", "slope", "box"])[
            int(lib.terrain_type_code[idx])],
        "anchor_mask": lib.anchor_mask_or_legacy()[idx:idx + 1],
        # Carry the source clip's REAL length (not the new common T) so a short clip padded into a
        # longer composed library still ends the episode at its own last frame.
        "clip_len": lib.clip_len_or_full()[idx:idx + 1].to(torch.int32),
    }
    if has_amt:
        b["anchor_mask_t"] = _pad(lib.anchor_mask_t[idx], T).unsqueeze(0)
    return b


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mix", required=True,
                   help="'path:tiles, ...' terrain sources (tiles = target tile count). A source whose "
                        "clips are all terrain_id -1 is treated as a flat pool (its tiles -> flat-tile count).")
    p.add_argument("--out", required=True)
    p.add_argument("--num-steps", type=int, default=None, help="Common clip length (default = max over sources).")
    p.add_argument("--flat-from", default=None,
                   help="Draw flat clips from this source's terrain_id==-1 clips (e.g. a stairs+flat library) "
                        "to size + fill the flat tiles. Flat tile SIZE = largest carried flat clip + buffer.")
    p.add_argument("--flat-tiles", type=int, default=None,
                   help="Number of flat tiles (default = sum of any --mix flat entries).")
    p.add_argument("--flat-clips", type=int, default=300, help="Flat clips to carry (evenly sampled) for the flat tiles.")
    p.add_argument("--max-clips-per-tile", type=int, default=0,
                   help="Cap clips carried per TERRAIN tile (0 = all). Use a small cap for a lightweight "
                        "dry-run/viz library; use 0 for a real training library.")
    p.add_argument("--types", default=None,
                   help="Comma-separated tile types to KEEP from each terrain source, e.g. 'terrain' for "
                        "pure stairs only, or 'terrain,transition'. Names come from the library's "
                        "terrain_type_names (flat/terrain/transition/slope/box). Default = keep all. Use "
                        "with --flat-tiles 0 to build a single-family library from an already-composed one.")
    p.add_argument("--device", default="cpu")
    # --- provenance. A composed library is the artifact everything downstream trains on, and until
    # now it recorded nothing about WHAT it was built from: only the deprecated motion-matching
    # generators ever wrote a source fingerprint, so the live Motion Bricks chain produced libraries
    # that could not answer "which solved library is this?". That matters now rather than in
    # principle -- the solved library is being replaced, and hf_cache never re-checks its local copy
    # against the remote, so two machines can hold different libraries under one name.
    p.add_argument("--library", default=None,
                   help="Solved traj-opt library this was stitched from (e.g. "
                        "trajectories/skills/library_v2). Fingerprinted into the output.")
    p.add_argument("--hf-repo", default="zolkin/robot_rl",
                   help="HuggingFace repo for --library; '' for a purely local path.")
    p.add_argument("--recipe", default=None,
                   help="clip-gen recipe YAML. Recorded VERBATIM as generator_config -- it is "
                        "already validated (no missing and no unknown keys), which is what makes it "
                        "worth recording rather than a hand-picked subset.")
    p.add_argument("--seed", type=int, default=None, help="Recipe seed, for the clip fingerprint.")
    args = p.parse_args()
    dev = args.device
    keep_types = {t.strip() for t in args.types.split(",")} if args.types else None

    specs = []
    for e in args.mix.split(","):
        path, tiles = e.strip().rsplit(":", 1)
        specs.append((path.strip(), int(tiles)))

    loaded = [(_load_source(path, keep_types), path, n) for path, n in specs]
    paths_for_T = [path for _, path, _ in loaded] + ([args.flat_from] if args.flat_from else [])
    _validate_meta(paths_for_T)                                      # before any clip is copied
    T = args.num_steps or max(int(_get_lib(path).num_steps) for path in paths_for_T)
    print(f"[compose] common num_steps = {T}")

    records, manifests, next_tid, flat_tiles, dist = [], [], 0, 0, {}
    flat_pool: list[tuple[str, int]] = []                            # (source_path, clip_idx)

    for (is_flat, tiles), path, n_target in loaded:
        name = path.split("/")[-1]
        if is_flat:
            flat_tiles += n_target
            flat_pool += [(path, i) for i in tiles[0][1]]
            continue
        # terrain source: replicate/subset its distinct tiles to n_target
        lib = _get_lib(path)
        picked_clips = 0
        for k in range(n_target):
            man_entry, clip_idx = tiles[k % len(tiles)]              # round-robin over the source's tiles
            keep = _sample(clip_idx, args.max_clips_per_tile)
            tid = next_tid; next_tid += 1
            m = dict(man_entry); m["terrain_id"] = tid; m["family"] = name   # family = source (jump/distractor/stairs)
            manifests.append(m)
            for i in keep:
                records.append(_clip_batch(lib, i, T, tid, dev))
            picked_clips += len(keep)
        dist[f"{name} (terrain)"] = (n_target, picked_clips)

    # Explicit flat source: its terrain_id==-1 clips size + fill the flat tiles.
    if args.flat_from:
        flib = _get_lib(args.flat_from)
        fmask = (flib.terrain_id.to(torch.long) == -1).nonzero().flatten().tolist()
        flat_pool += [(args.flat_from, i) for i in fmask]
    if args.flat_tiles is not None:
        flat_tiles = args.flat_tiles

    n_flat_carried = 0
    if flat_pool and flat_tiles > 0:                                 # carry an evenly-sampled subset of the pool
        for j in _sample(list(range(len(flat_pool))), args.flat_clips):
            src, i = flat_pool[j]
            records.append(_clip_batch(_get_lib(src), i, T, -1, dev))
        n_flat_carried = min(len(flat_pool), args.flat_clips) if args.flat_clips > 0 else len(flat_pool)

    total_tiles = next_tid + flat_tiles
    print(f"\n[compose] === training tile distribution ({total_tiles} tiles, envs round-robin over them) ===")
    for name, (nt, nc) in dist.items():
        print(f"  {name:44s}: {nt:4d} tiles ({100*nt/max(1,total_tiles):5.1f}%)  {nc} clips")
    flat_srcs = ",".join(sorted({s.split('/')[-1] for s, _ in flat_pool})) or "none"
    print(f"  {'flat ground':44s}: {flat_tiles:4d} tiles ({100*flat_tiles/max(1,total_tiles):5.1f}%)  "
          f"{n_flat_carried} clips from [{flat_srcs}]")

    meta = dict(_get_lib(specs[0][0]).meta)
    meta["num_steps"] = int(T)
    meta["clip_length_s"] = T * meta["dt"]
    writer = ClipLibraryWriter(args.out, len(records), meta)
    writer.set_attr("generator", "composed")
    writer.set_attr("terrain_manifest", manifests)
    _sources = [pth for pth, _n in specs] + ([args.flat_from] if args.flat_from else [])

    # generator_config is the WHOLE recipe when one was given, with the flat-tile count folded in
    # (the loader reads counts.flat, so it cannot move). The recipe is a validated dict by the time
    # it gets here -- clip_gen_config guarantees no missing and no unknown keys -- so recording it
    # verbatim is strictly better than a curated subset that drifts from what was actually used.
    gen_cfg: dict = {"counts": {"flat": flat_tiles}}
    if args.recipe:
        import yaml
        with open(args.recipe) as f:
            gen_cfg["recipe"] = yaml.safe_load(f)
        gen_cfg["recipe_path"] = os.path.basename(args.recipe)
    writer.set_attr("generator_config", gen_cfg)

    # If a SOURCE family records its own solved-library fingerprint, it must agree with the one
    # being stamped here. Nothing does yet -- the family writers are not plumbed for it -- so this
    # is inert today and is written for the case it exists to catch: `--extra-mix`, which folds in
    # families built by a separate invocation, possibly weeks apart and possibly against a
    # different solved library. Composing those silently would produce a library whose recorded
    # provenance is true of only some of its clips.
    # TODO(family-provenance): have mb_assemble_library / mb_pure_stair_clips / mb_flat_clips
    # record source_library_fingerprint so this check has something to compare.
    _src_fps = {pth: (_get_lib(pth).attrs or {}).get("source_library_fingerprint")
                for pth in _sources}
    _seen = {json.dumps(v, sort_keys=True) for v in _src_fps.values() if v}
    if len(_seen) > 1:
        print("[compose][WARN] sources disagree about which solved library they came from:")
        for pth, v in _src_fps.items():
            if v:
                print(f"[compose][WARN]   {os.path.basename(pth)}: {v.get('short', '?')} "
                      f"({v.get('source', '?')})")

    src_fp: dict = {}
    if args.library:
        src_fp = fingerprint_solved_library(args.library, args.hf_repo or None)
        writer.set_attr("source_library_fingerprint", src_fp)
        print(f"[compose] solved library: {src_fp['source']} {src_fp['short']} "
              f"({src_fp['num_clips']} clips)")
    else:
        print("[compose][WARN] no --library given: the output will not record which solved library "
              "it was built from, so 'is this the library I think it is?' becomes unanswerable.")
    # The one fingerprint that ties all three together. fingerprint_clip_config has existed since
    # the clip machinery was written and was, in its own words, "imported in one place and called in
    # none, so the existing check has never had anything to compare".
    writer.set_attr("clip_fingerprint",
                    fingerprint_clip_config(src_fp, gen_cfg, args.seed))
    # Carry the REGIME declaration through. ``stride_idx >= 0`` marks the exact looped-replay frames, and
    # QC needs an explicit attr to trust it because the dataset's h5py fill is 0 -- a library that never
    # wrote it would otherwise read as "stride 0 everywhere", i.e. 100% steady. Propagated when ANY source
    # declares it: every other family writes -1, so their frames are already correctly marked "not a loop".
    # ``--flat-from`` is a separate input from ``--mix`` and is the family that actually carries the
    # regime, so it has to be checked too -- missing it left the composed library unscoreable.
    if any(bool(_get_lib(pth).attrs.get("has_regime", False)) for pth in _sources):
        writer.set_attr("has_regime", True)
    for i, b in enumerate(records):
        writer.write_batch(i, b)
    writer.close()

    lib = load_clip_library(args.out)
    yy = torch.as_tensor(lib.y_des)
    print(f"\n[compose] wrote {args.out}: {lib.num_clips} clips x {lib.num_steps} steps, "
          f"{total_tiles} tiles, finite={bool(torch.isfinite(yy).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
