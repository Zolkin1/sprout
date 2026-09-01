# Copyright 2026 Zachary Olkin. All rights reserved.

"""What would a perfect picker of the generator's noise draw be worth?

Reads the ``cells_seed<k>`` produced by running one plan under many ``--fm_seed`` values with
everything else pinned, and answers one question per family: if you could choose the seed per
clip, how much more often would the robot arrive?

**The number is reported against a SHUFFLE, and only the difference means anything.**  With
``S`` seeds and ``R`` robots, ``max_k`` over the per-seed rates is biased upward even when the
seed does nothing at all -- ten robots wobble, and taking a maximum over sixteen wobbles finds
a high one.  So the same statistic is recomputed on the same episodes randomly repartitioned
into pseudo-seeds, which destroys any seed effect while preserving the noise exactly.  Real
minus shuffled is the seed-attributable gain, and the shuffle distribution gives its p-value
without assuming anything about the noise.

This is the same discipline as the harness's same-noise control, applied at episode scale:
the previous version of this project died because a control was not run, and its headline
turned out to be its own measurement noise.

Imports no simulator.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import report as eval_report  # noqa: E402
import schema  # noqa: E402


def load_cell(path: str) -> dict:
    """Per-episode success, clip and family for one cell."""
    d, meta = schema.read_episode_table(os.path.join(path, "episodes.h5"))
    terms = list(meta.get("termination_terms") or [])
    _, fail_terms = eval_report.term_partition(meta)
    bits = d["termination_bits"]
    failed = np.zeros(bits.shape, dtype=bool)
    for name in fail_terms:
        if name in terms:
            failed |= (bits & (1 << terms.index(name))) != 0
    names = {int(v): k for k, v in (meta.get("family_codes") or {}).items()}
    return {"ok": d["reached_goal"].astype(bool) & ~failed,
            "clip": np.asarray(d["clip_id"]),
            "fam": np.array([names.get(int(c), "?") for c in d["family_code"]])}


def gain(ok_by_seed: np.ndarray) -> tuple[float, float]:
    """``(current, attainable)`` for one clip.

    Args:
        ok_by_seed: ``[S, R]`` boolean outcomes, seeds by robots.

    Returns:
        The mean success rate, and the best seed's rate.
    """
    per_seed = ok_by_seed.mean(axis=1)
    return float(per_seed.mean()), float(per_seed.max())


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="run dir holding cells_seed*/")
    p.add_argument("--cell", default="fm__none__nojumpoff")
    p.add_argument("--shuffles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cells = sorted(
        (int(re.search(r"cells_seed(\d+)$", d).group(1)), os.path.join(args.dir, d, args.cell))
        for d in os.listdir(args.dir)
        if re.match(r"^cells_seed\d+$", d)
        and os.path.isdir(os.path.join(args.dir, d, args.cell)))
    if len(cells) < 2:
        raise SystemExit(f"need at least 2 seed cells in {args.dir}; found {len(cells)}")
    print(f"seeds: {[k for k, _ in cells]}")

    per = [load_cell(path) for _, path in cells]
    clips = sorted(set(per[0]["clip"].tolist()))
    fam_of = {int(c): per[0]["fam"][per[0]["clip"] == c][0] for c in clips}

    # [clip][seed][robot]; robot order within a clip is arbitrary but consistent in length.
    rows, keep = [], []
    for c in clips:
        block = [d["ok"][d["clip"] == c] for d in per]
        n = min(len(b) for b in block)
        if n == 0:
            continue
        rows.append(np.stack([b[:n] for b in block]))
        keep.append(c)
    rng = np.random.default_rng(args.seed)

    def summarise(sel) -> tuple[float, float]:
        cur = att = 0.0
        for m in sel:
            a, b = gain(m)
            cur += a
            att += b
        return cur / len(sel), att / len(sel)

    print(f"\n{'family':<14} {'n':>4} {'current':>8} {'attainable':>11} {'raw gain':>9} "
          f"{'shuffled':>9} {'NET':>8} {'p':>7}")
    for fam in sorted(set(fam_of.values())) + ["ALL"]:
        sel = [m for m, c in zip(rows, keep) if fam == "ALL" or fam_of[c] == fam]
        if not sel:
            continue
        cur, att = summarise(sel)
        # The SHUFFLE: repartition each clip's own episodes into the same seed/robot shape,
        # which removes the seed effect and keeps the noise.
        null = np.empty(args.shuffles)
        for i in range(args.shuffles):
            sh = []
            for m in sel:
                flat = m.reshape(-1).copy()
                rng.shuffle(flat)
                sh.append(flat.reshape(m.shape))
            c2, a2 = summarise(sh)
            null[i] = a2 - c2
        raw = att - cur
        net = raw - float(null.mean())
        pval = float((null >= raw).mean())
        print(f"{fam:<14} {len(sel):>4} {cur:>8.3f} {att:>11.3f} {raw:>+9.3f} "
              f"{null.mean():>+9.3f} {net:>+8.3f} {pval:>7.3f}")
    print("\nNET is the seed-attributable gain: what a perfect per-clip seed picker buys "
          "beyond\nwhat taking a maximum over noisy estimates buys by itself. It is the "
          "effect size any\nfuture selector A/B has to be powered to detect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
