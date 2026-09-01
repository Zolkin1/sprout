# Copyright 2026 Zachary Olkin. All rights reserved.

"""Merge per-cell selected-chunk recordings into one filtered-BC training file.

Each cell of the collection run wrote ``selected.h5`` (the chunk the critic committed at every
re-plan, with its conditioning) beside ``live_select.json`` (one row per episode, carrying the
verdict).  This joins them on ``episode_uid`` and writes a single file for
``finetune_generator.py``.

**Two filters, and they answer different questions.**  The recording is already filtered by
SELECTION -- rejected candidates were never written -- so a merge with ``--all`` trains on
"reproduce the chunk the critic chose".  Adding ``--successful_only`` further keeps a chunk
only if the episode it belonged to ended in success, i.e. "reproduce the chunk the critic
chose, in runs that worked".

**What outcome-filtering does to the mix, stated because it is not neutral.**  K=4 success is
strongly speed-dependent (~0.97 at 0.8 m/s against ~0.65 at 2.4), so dropping failed episodes
removes far more fast data than slow.  The surviving set is therefore skewed toward the easy
regime -- the opposite of where the headroom is.  The per-speed retention printed below is the
number to read before trusting a fine-tune from it.

**Survivorship, in both directions.**  A chunk from a failed episode is not necessarily a bad
chunk (the failure may have come later, or elsewhere), and a chunk inside a successful episode
is not necessarily good.  Episode-level outcome is a coarse label for a per-re-plan decision;
that is the price of not having a per-chunk one.

Usage::

    python merge_selected.py --root logs/value/bc_data_v1 \\
        --out logs/value/bc_data_v1/train_success.h5 --successful_only
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

#: Columns every recording carries; the only ones ``rollout_dataset`` reads.
COLUMNS = ("episode_uid", "replan_index", "family_code", "history", "history_null",
           "extero", "cmd_future", "fm_plan")

#: Present only with ``live_select.py --record_candidates``.  DIAGNOSTIC columns -- the
#: trainer never reads them; ``ft_canary.PairGapCanary`` does.  All-or-nothing across cells:
#: a merge that quietly kept the intersection would produce a file whose candidate coverage
#: depends on which cells happened to finish, and the paired statistic would then be computed
#: over a silently different subset than the training data.
CANDIDATE_COLUMNS = ("cand_plans", "cand_scores", "cand_verdict", "sel_index")

#: Derived here rather than left to the consumer.  ``--successful_only`` already resolves every
#: row's episode outcome to decide what to keep; writing it down costs one byte a row and means
#: a downstream metric can stratify by it WITHOUT re-reading 63 ``live_select.json`` files and
#: re-deriving the uid encoding -- a second implementation of a join is a second thing to get
#: wrong.  On a ``--successful_only`` file it is constant 1, which is correct and harmless.
OUTCOME_COLUMN = "episode_success"


def episode_verdicts(cell: str) -> dict[int, str]:
    """Map ``episode_uid`` -> verdict for the K>1 arm of one cell.

    Rebuilds the uid with the same encoding ``live_select.run`` used when recording, so the
    join is on a key both sides compute the same way rather than on row order.

    Args:
        cell: The cell directory holding ``live_select.json``.

    Returns:
        ``{episode_uid: verdict}``; empty when the json is missing.
    """
    path = os.path.join(cell, "live_select.json")
    if not os.path.isfile(path):
        return {}
    blob = json.load(open(path))
    speed = float(blob.get("args", {}).get("speed", 0.0))
    out = {}
    for r in blob["rows"]:
        if int(r["k"]) <= 1:
            continue                      # K=1 arms record nothing; nothing to join to
        uid = (int(r["tile"]) * 1000000 + int(r["k"]) * 100000
               + int(round(speed * 10)) * 1000 + int(r["episode"]))
        out[uid] = r["verdict"]
    return out


def main() -> int:
    """Entry point."""
    import h5py

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="collection root holding t<tile>_s<speed>/")
    p.add_argument("--out", required=True)
    p.add_argument("--successful_only", action="store_true",
                   help="Keep a chunk only if its episode ended in success.")
    args = p.parse_args()

    cells = sorted(d for d in os.listdir(args.root)
                   if os.path.isfile(os.path.join(args.root, d, "selected.h5")))
    if not cells:
        raise SystemExit(f"no cells with selected.h5 under {args.root}")

    # All-or-nothing, decided BEFORE any data is read, so a partial collection fails here
    # rather than producing a file that looks complete.
    have = []
    for d in cells:
        with h5py.File(os.path.join(args.root, d, "selected.h5"), "r") as f:
            have.append(bool(f.attrs.get("has_candidates", 0)))
    if any(have) and not all(have):
        missing = [d for d, h in zip(cells, have) if not h]
        raise SystemExit(
            f"{sum(have)}/{len(cells)} cells carry candidate columns; these do not: "
            f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}\n"
            "Merging anyway would drop them everywhere (making the pair metric impossible) or "
            "keep an intersection whose coverage depends on which cells finished. Re-run the "
            "missing cells with --record_candidates, or delete them.")
    cand = all(have) and bool(have)
    columns = tuple(COLUMNS) + (CANDIDATE_COLUMNS if cand else ())
    print(f"[merge] candidate columns: {'PRESENT' if cand else 'absent'}")

    columns = columns + (OUTCOME_COLUMN,)
    parts: dict[str, list] = {c: [] for c in columns}
    n_seen = n_kept = 0
    by_speed: dict[float, list[int]] = {}

    for d in cells:
        cell = os.path.join(args.root, d)
        verdicts = episode_verdicts(cell)
        speed = float(d.rsplit("_s", 1)[-1])
        with h5py.File(os.path.join(cell, "selected.h5"), "r") as f:
            n = int(f.attrs["num_windows_written"])
            uid = f["episode_uid"][:n]
            keep = np.ones(n, dtype=bool)
            if args.successful_only:
                keep = np.array([verdicts.get(int(u)) == "success" for u in uid])
            n_seen += n
            n_kept += int(keep.sum())
            by_speed.setdefault(speed, [0, 0])
            by_speed[speed][0] += n
            by_speed[speed][1] += int(keep.sum())
            if not keep.any():
                continue
            ok = np.array([verdicts.get(int(u)) == "success" for u in uid], dtype=np.uint8)
            for c in columns:
                col = ok if c == OUTCOME_COLUMN else f[c][:n]
                parts[c].append(col[keep])

    merged = {c: np.concatenate(parts[c]) for c in columns}
    total = merged["episode_uid"].shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as f:
        for c in columns:
            f.create_dataset(c, data=merged[c])
        f.attrs["num_windows_written"] = total
        f.attrs["is_selected_chunk_recording"] = 1
        f.attrs["has_candidates"] = int(cand)
        f.attrs["filter"] = "selected+successful" if args.successful_only else "selected"
        f.attrs["n_recorded"] = n_seen
        f.attrs["n_cells"] = len(cells)
        f.attrs["columns_written"] = json.dumps(list(columns))

    print(f"[merge] {len(cells)} cells | recorded {n_seen:,} -> kept {total:,} "
          f"({total / max(n_seen, 1):.1%})"
          + ("  [successful episodes only]" if args.successful_only else "  [all selected]"))
    print(f"[merge] {len(np.unique(merged['episode_uid'])):,} episodes survive")
    print("[merge] retention by speed -- outcome-filtering is NOT speed-neutral:")
    for sp in sorted(by_speed):
        seen, kept = by_speed[sp]
        print(f"    {sp:>4} m/s   {kept:>7,} / {seen:>7,}   {kept / max(seen, 1):.1%}")
    print(f"[merge] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
