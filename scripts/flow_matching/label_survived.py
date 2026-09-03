#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""The CONTROL label for filtered BC: every chunk of every episode that survived.

This needs no critic, and that is the point.  A filtered-BC fine-tune on an advantage label
has to beat the rule anyone would write down first -- keep the successful episodes -- or the
critic contributed nothing to it.  Running the two arms from the same recording, through the
same trainer, differing only in which sidecar selects the rows, is what makes the comparison
attributable.

The sidecar it writes is byte-compatible with ``label_advantage.py``'s, so
``finetune_generator.py --labels <this> --filter_label positive`` needs no other change, and
``inspect_filter.py`` can put the two arms in one table.

**It is written from ``predictions.npz``, not from the episode table**, so it inherits exactly
the row set the advantage label was built on -- the same censored and post-latch drops.  Built
from the episode table instead, the control would train on rows the treatment never saw and
the difference between them would include that.

Usage::

    python scripts/flow_matching/label_survived.py \\
        logs/value/v4/predictions.npz --out logs/value/<run>/survived_only.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "source", "robot_rl"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_rl.flow_matching import advantage as adv_mod  # noqa: E402


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", help="predictions.npz from fit_value.py")
    p.add_argument("--out", required=True, help="Output .npz sidecar.")
    p.add_argument("--exclude_tile_holdout", action="store_true",
                   help="Drop windows on held-out tiles (they are the generalisation "
                        "measure; training on them spends it).")
    args = p.parse_args()

    d = np.load(args.predictions)
    uid, ridx, fam = d["episode_uid"], d["replan_index"], d["family_code"]
    failed, val_tile = d["failed"].astype(bool), d["val_tile"].astype(bool)

    labels = np.where(failed, adv_mod.LABEL_NEGATIVE, adv_mod.LABEL_POSITIVE).astype(np.int64)
    if args.exclude_tile_holdout:
        labels = np.where(val_tile, adv_mod.LABEL_NULL, labels)

    counts = {name: int((labels == i).sum()) for i, name in enumerate(adv_mod.LABEL_NAMES)}
    n_ep = int(np.unique(uid).size)
    n_ep_pos = int(np.unique(uid[labels == adv_mod.LABEL_POSITIVE]).size)
    print(f"[survived] {counts}  over {n_ep} episodes, {n_ep_pos} of them contributing")
    print(f"[survived] positive fraction {float((labels == adv_mod.LABEL_POSITIVE).mean()):.3f}")

    meta = {"rule": "every chunk of every episode that did not fail",
            "critic": None, "predictions": os.path.abspath(args.predictions),
            "exclude_tile_holdout": bool(args.exclude_tile_holdout), "counts": counts}
    # `advantage` is the label itself as a float, so the file has the same columns as
    # `label_advantage.py`'s and nothing downstream needs a special case. There is no
    # advantage here to record -- the rule reads only the outcome.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, label=labels.astype(np.int8),
             advantage=(~failed).astype(np.float32),
             bootstrapped=np.ones(uid.size, dtype=bool),
             episode_uid=uid, replan_index=ridx, family_code=fam,
             val_tile=val_tile, meta=json.dumps(meta))
    print(f"[survived] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
