#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""What does a filtered-BC label actually SELECT?  Run this before spending a fine-tune on it.

A percentile cut states what fraction of rows it keeps and nothing about WHICH rows, and every
way it can go wrong leaves the fraction correct.  A cut that drops whole episodes trains on a
biased slice of the rollout; one that keeps only slow commands teaches a policy that has never
been asked to move; one that abandons tiles removes the terrain the fine-tune exists to fix.
None of that is visible in the label counts.

So this reports the composition of the kept set along every axis the recording carries -- and
beside it the composition of the WHOLE recording, because a share is only readable against the
share it started from.  A ``--baseline`` sidecar (the survived-episodes-only control, say) is
compared column for column, so "what does the critic buy over the trivial rule" is answered in
the same table rather than by holding two outputs side by side.

**The speeds come from the recording, not the label**, so this needs the rollout ``.h5``:
``cmd_future`` is the commanded velocity horizon, and its first row is the command in force at
the window.  Everything else joins from the sidecar.

Usage::

    python scripts/flow_matching/inspect_filter.py \\
        --labels logs/value/<run>/advantage_amc_top30.npz \\
        --baseline logs/value/<run>/survived_only.npz \\
        --rollout logs/rollouts/v4_big/rollout_<...> \\
        --out logs/value/<run>/filter_report.md
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

#: Family codes as the rollout's ``family_names`` attr orders them.
FAMILY_NAMES = ("flat/unknown", "stairs_up", "stairs_down", "jump_on", "jump_off_front")


def _progress(uid: np.ndarray, ri: np.ndarray) -> np.ndarray:
    """Normalized position in the episode, from the table's OWN last index.

    Args:
        uid: ``[W]`` episode ids.
        ri: ``[W]`` re-plan index.

    Returns:
        ``[W]`` in ``[0, 1]``.
    """
    _, inv = np.unique(uid, return_inverse=True)
    last = np.zeros(uid.size, dtype=np.int64)
    np.maximum.at(last, inv, ri)
    denom = np.maximum(last[inv], 1)
    return np.where(last[inv] > 0, ri / denom, 0.0)


def _bucket_table(name: str, keys: np.ndarray, labels, keep: dict) -> list[str]:
    """One composition table: the population's share, then each arm's share and keep rate.

    Args:
        name: Axis name for the header.
        keys: ``[W]`` bucket id per row.
        labels: Bucket id -> display name, or ``None`` to use the id.
        keep: Arm name -> ``[W]`` bool mask.

    Returns:
        Markdown lines.
    """
    uniq = np.unique(keys)
    arms = list(keep)
    head = ["| " + name + " | rows | share of all |"
            + "".join(f" {a}: share of keeps | {a}: keep rate |" for a in arms),
            "|---|---|---|" + "---|---|" * len(arms)]
    out = head
    for u in uniq:
        s = keys == u
        disp = labels[u] if labels is not None and u in labels else str(u)
        cells = [f"{disp}", f"{int(s.sum())}", f"{s.mean():.3f}"]
        for a in arms:
            k = keep[a]
            cells += [f"{(k & s).sum() / max(k.sum(), 1):.3f}", f"{k[s].mean():.3f}"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="The label sidecar to inspect.")
    p.add_argument("--baseline", default="", help="A second sidecar to compare against.")
    p.add_argument("--rollout", required=True, help="Rollout stem, for commands and tiles.")
    p.add_argument("--out", required=True, help="Markdown report to write.")
    p.add_argument("--speed_bins", type=float, nargs="+",
                   default=[0.0, 0.2, 0.5, 0.9, 1.3, 99.0],
                   help="Edges of the commanded-speed bins, in m/s.")
    args = p.parse_args()

    import h5py

    L = np.load(args.labels, allow_pickle=True)
    uid, ri, fam = L["episode_uid"], L["replan_index"], L["family_code"]
    keep = {"A_MC": L["label"] == 1}
    meta = {"labels": json.loads(str(L["meta"])) if "meta" in L.files else {}}
    if args.baseline:
        B = np.load(args.baseline, allow_pickle=True)
        if not (B["episode_uid"] == uid).all() or not (B["replan_index"] == ri).all():
            raise SystemExit(
                "the two sidecars are not row-aligned, so every comparison below would join "
                "the wrong rows. Both must come from the same predictions table.")
        keep["survived"] = B["label"] == 1
        meta["baseline"] = json.loads(str(B["meta"])) if "meta" in B.files else {}

    # ---- pull the axes the sidecar does not carry -------------------------------------
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
    with h5py.File(args.rollout + ".h5", "r") as f:
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        cmd0 = f["cmd_future"][:n, 0, :]        # the command in force at this window
        w_uid, w_ri = f["episode_uid"][:n], f["replan_index"][:n]
        clip = f["clip_id"][:n]
    with h5py.File(args.rollout + ".episodes.h5", "r") as f:
        ep_uid, ep_tile = f["episode_uid"][:], f["terrain_id"][:]
        ep_failed = f["terminated"][:].astype(bool)

    # Join the recording onto the sidecar by KEY. The sidecar dropped censored and
    # post-latch rows, so the two tables differ in length and a positional join is wrong.
    key_w = w_uid.astype(np.int64) * 100000 + w_ri.astype(np.int64)
    key_l = uid.astype(np.int64) * 100000 + ri.astype(np.int64)
    order = np.argsort(key_w)
    pos = order[np.searchsorted(key_w, key_l, sorter=order)]
    if not (key_w[pos] == key_l).all():
        raise SystemExit("some sidecar rows have no matching window in the rollout; the two "
                         "do not describe the same run.")
    speed = np.linalg.norm(cmd0[pos][:, :2], axis=1)
    clip_l = clip[pos]
    tile_of = dict(zip(ep_uid.tolist(), ep_tile.tolist()))
    fail_of = dict(zip(ep_uid.tolist(), ep_failed.tolist()))
    tile = np.array([tile_of.get(int(u), -999) for u in uid])
    failed = np.array([fail_of.get(int(u), False) for u in uid])
    prog = _progress(uid, ri)

    lines = [f"# Filter composition -- `{os.path.basename(args.labels)}`\n",
             f"Rollout `{args.rollout}`. Sidecar rows {uid.size}, episodes "
             f"{np.unique(uid).size}, tiles {np.unique(tile).size}, clips "
             f"{np.unique(clip_l).size}.\n"]
    for a, k in keep.items():
        lines.append(f"- **{a}**: {int(k.sum())} rows ({k.mean():.1%}), "
                     f"{np.unique(uid[k]).size} episodes, {np.unique(tile[k]).size} tiles, "
                     f"{np.unique(clip_l[k]).size} clips\n")

    # ---- the coverage question, asked directly ----------------------------------------
    lines.append("\n## Coverage: does every episode / tile / clip survive the cut?\n")
    rows = []
    all_ep, all_tile, all_clip = np.unique(uid), np.unique(tile), np.unique(clip_l)
    for a, k in keep.items():
        n_ep = np.unique(uid[k]).size
        per_ep = np.bincount(np.unique(uid, return_inverse=True)[1], weights=k.astype(float))
        rows.append([a, f"{n_ep}/{all_ep.size}", f"{all_ep.size - n_ep}",
                     f"{np.unique(tile[k]).size}/{all_tile.size}",
                     f"{np.unique(clip_l[k]).size}/{all_clip.size}",
                     f"{np.percentile(per_ep, 10):.0f}", f"{np.median(per_ep):.0f}",
                     f"{np.percentile(per_ep, 90):.0f}"])
    lines += ["| arm | episodes kept | episodes LOST | tiles | clips | chunks/ep p10 | median | p90 |",
              "|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]

    lines.append("\n## Family\n")
    lines += _bucket_table("family", fam,
                           {i: f"{i} {n}" for i, n in enumerate(FAMILY_NAMES)}, keep)
    lines.append("\n## Progress through the episode\n")
    pb = np.clip((prog * 5).astype(int), 0, 4)
    lines += _bucket_table("progress", pb,
                           {i: f"{i/5:.1f}-{(i+1)/5:.1f}" for i in range(5)}, keep)
    lines.append("\n## Commanded speed (|v_xy| at the window)\n")
    sb = np.clip(np.digitize(speed, args.speed_bins) - 1, 0, len(args.speed_bins) - 2)
    lines += _bucket_table("speed m/s", sb,
                           {i: f"{args.speed_bins[i]:.1f}-{args.speed_bins[i+1]:.1f}"
                            for i in range(len(args.speed_bins) - 1)}, keep)
    lines.append("\n## Realized episode outcome\n")
    lines += _bucket_table("outcome", failed.astype(np.int64),
                           {0: "survived", 1: "failed"}, keep)

    # ---- per-tile keep rate, the axis most likely to collapse quietly -----------------
    lines.append("\n## Tiles: the extremes\n")
    for a, k in keep.items():
        rate = np.array([k[tile == t].mean() for t in all_tile])
        cnt = np.array([int((tile == t).sum()) for t in all_tile])
        o = np.argsort(rate)
        lo = ", ".join(f"{all_tile[i]}:{rate[i]:.2f}(n={cnt[i]})" for i in o[:5])
        hi = ", ".join(f"{all_tile[i]}:{rate[i]:.2f}(n={cnt[i]})" for i in o[-5:])
        lines.append(f"- **{a}** keep rate: min {rate.min():.3f}, max {rate.max():.3f}, "
                     f"sd {rate.std():.3f}\n  - lowest: {lo}\n  - highest: {hi}\n")

    if len(keep) > 1:
        a, b = list(keep)
        lines.append(f"\n## `{a}` against `{b}`\n")
        ka, kb = keep[a], keep[b]
        lines.append(f"- rows in both: {int((ka & kb).sum())}; only `{a}`: "
                     f"{int((ka & ~kb).sum())}; only `{b}`: {int((~ka & kb).sum())}\n")
        lines.append(f"- of `{a}`'s keeps, {float((ka & failed).sum() / max(ka.sum(),1)):.3f} "
                     f"come from episodes that FAILED\n")
        lines.append(f"- `{b}` keeps {float(kb.sum() / max(ka.sum(), 1)):.2f}x as many rows\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[inspect] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
