# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compare two advantage labellings of ONE rollout -- what does the cross-fit actually buy?

Built for the ``--folds 2`` vs ``--folds 1`` comparison, but it takes any two sidecars over
the same windows (two look-aheads, two percentiles, two critics).

The hypothesis under test is specific.  An in-sample critic can memorise the outcome of the
very episode it is scoring, and every window of an episode carries the SAME binary label, so
the effective sample size is the episode count and not the window count.  Under memorisation
``V(s_t)`` and ``V(s_{t+n})`` both collapse onto the realized outcome, their difference goes
to ZERO, and the advantage stops carrying information -- while remaining a perfectly
well-formed array of the right shape, dtype and class balance.

So the diagnostic is not "do the labels differ" alone.  Three things are printed and they can
disagree with each other:

* **agreement** on the shared windows.  Near 1.0 means the fold structure changed nothing here.
* **the advantage SPREAD** per arm.  Collapse shows up as the in-sample arm having a
  materially smaller ``sd`` and interquartile range than the cross-fit one -- that is the
  memorisation signature, and it is visible without any downstream run.
* **the G2 band deltas** carried in each sidecar's ``meta``.  The in-sample arm's are
  inflated by construction (the critic partly knows the answer), so a HIGHER G2 there is
  expected and is not evidence that it is the better label.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

_NAMES = ("negative", "positive", "null")


def _key(uid: np.ndarray, ridx: np.ndarray) -> np.ndarray:
    """Pack ``(episode_uid, replan_index)`` into one sortable int64 key."""
    return (uid.astype(np.int64) << np.int64(20)) + ridx.astype(np.int64)


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="First advantage.npz.")
    p.add_argument("b", help="Second advantage.npz.")
    p.add_argument("--names", nargs=2, default=["a", "b"], help="Labels for the printout.")
    args = p.parse_args()

    da, db = np.load(args.a), np.load(args.b)
    na, nb = args.names
    ka, kb = _key(da["episode_uid"], da["replan_index"]), _key(db["episode_uid"],
                                                               db["replan_index"])
    shared, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    print(f"[cmp] {na}: {ka.size} windows   {nb}: {kb.size}   shared: {shared.size}")
    if shared.size == 0:
        raise SystemExit("the two sidecars share no windows -- are they from one rollout?")

    la, lb = da["label"][ia], db["label"][ib]
    aa, ab = da["advantage"][ia], db["advantage"][ib]

    agree = float((la == lb).mean())
    print(f"[cmp] label agreement on shared windows: {agree:.4f}")
    print("[cmp] confusion (rows = {}, cols = {}):".format(na, nb))
    print("           " + "".join(f"{n:>12s}" for n in _NAMES))
    for i, n in enumerate(_NAMES):
        row = [int(((la == i) & (lb == j)).sum()) for j in range(3)]
        print(f"    {n:>7s}" + "".join(f"{v:12d}" for v in row))

    print("\n[cmp] advantage spread -- COLLAPSE is the in-sample failure mode, and it shows "
          "here\n      as a materially SMALLER spread, not as a different mean:")
    for name, v in ((na, aa), (nb, ab)):
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        print(f"    {name:>12s}  mean {v.mean():+.4f}  sd {v.std():.4f}  IQR {iqr:.4f}  "
              f"|A|>0.05 on {float((np.abs(v) > 0.05).mean()):.1%}")
    r = float(np.corrcoef(aa, ab)[0, 1])
    print(f"    correlation between the two advantages: {r:.4f}")

    print("\n[cmp] G2 median band delta, from each sidecar's own meta. The in-sample arm's is")
    print("      INFLATED by construction -- its critic partly knows the outcome -- so a")
    print("      higher number there is expected and is NOT evidence it is the better label:")
    for name, d in ((na, da), (nb, db)):
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        deltas = [b["delta"] for b in meta.get("g2_bands", []) if b.get("delta") is not None]
        folds = meta.get("num_folds", "?")
        if deltas:
            print(f"    {name:>12s}  folds={folds}  median {float(np.median(deltas)):+.3f}  "
                  f"({sum(x > 0 for x in deltas)}/{len(deltas)} bands positive)")
        else:
            print(f"    {name:>12s}  folds={folds}  no reportable bands")

    print("\n[cmp] READING IT: if agreement is high AND the spreads match, the cross-fit "
          "bought\n      nothing on this data and the cheaper fit is fine. If the in-sample "
          "spread is\n      much smaller, that is the collapse, and its label is noise "
          "wearing the right shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
