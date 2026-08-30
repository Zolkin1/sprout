# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn a critic's per-window ``P(fail)`` into the advantage label the fine-tune conditions on.

Reads ``predictions.npz`` (from ``fit_value.py``) and writes ``advantage.npz`` -- a sidecar
keyed by window index in the SAME order as the rollout ``.h5``, so the fine-tune dataset
joins it on ``(episode_uid, replan_index)`` -- see ``rollout_dataset.join_labels`` for why a
positional join would be silently wrong.

Also runs the plan's **G2 gate**, which is the load-bearing one and the reason this is a
separate stage rather than a line inside the trainer::

    within narrow bands of V(s_t), do positive-labelled chunks fail less than
    negative-labelled ones?

G1 ("the critic predicts failure") can pass by a wide margin while G2 fails completely --
that happens exactly when the advantage is reproducing state difficulty, in which case the
label carries no information about the CHUNK and conditioning on it cannot work.  Banding on
``V(s_t)`` is what controls for difficulty; the unbanded comparison is true by construction
and says nothing.

Usage::

    python scripts/flow_matching/label_advantage.py \\
        logs/value/v2/predictions.npz --rollout logs/rollouts/v2/rollout_<...> \\
        --n 8 --out logs/value/v2/advantage.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "source", "motion_gen"),
           os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_rl.flow_matching import advantage as adv_mod  # noqa: E402


def _bandwise_gate(p_state: np.ndarray, labels: np.ndarray, failed: np.ndarray,
                   n_bands: int = 5) -> list[dict]:
    """G2: compare positive- and negative-labelled chunks WITHIN narrow ``V(s_t)`` bands.

    Args:
        p_state: ``[W]`` ``P(fail | s_t)``.
        labels: ``[W]`` in ``{0, 1, 2}``.
        failed: ``[W]`` bool, the realized episode outcome.
        n_bands: Number of equal-count bands over ``p_state``.

    Returns:
        One record per band: its ``P(fail)`` range, the counts, the realized failure rate
        under each label and their difference.  A band with fewer than 50 of either label
        reports ``None`` for the rates rather than a number off a handful of episodes.
    """
    edges = np.quantile(p_state, np.linspace(0.0, 1.0, n_bands + 1))
    edges[-1] = np.inf
    out = []
    for i in range(n_bands):
        sel = (p_state >= edges[i]) & (p_state < edges[i + 1])
        pos = sel & (labels == adv_mod.LABEL_POSITIVE)
        neg = sel & (labels == adv_mod.LABEL_NEGATIVE)
        rec = {"band": i, "p_lo": float(edges[i]),
               "p_hi": float(edges[i + 1]) if np.isfinite(edges[i + 1]) else 1.0,
               "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        if pos.sum() >= 50 and neg.sum() >= 50:
            rec["fail_pos"] = float(failed[pos].mean())
            rec["fail_neg"] = float(failed[neg].mean())
            rec["delta"] = rec["fail_neg"] - rec["fail_pos"]
        else:
            rec["fail_pos"] = rec["fail_neg"] = rec["delta"] = None
        out.append(rec)
    return out


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", help="predictions.npz from fit_value.py")
    p.add_argument("--out", required=True, help="Output .npz sidecar.")
    p.add_argument("--variant", default="V_priv", choices=("V_priv", "V_onboard"),
                   help="Which critic to label with.")
    p.add_argument("--n", type=int, default=8,
                   help="Advantage look-ahead in re-plans (8 = 1.92 s at K=12, 50 Hz).")
    p.add_argument("--percentile", type=float, default=35.0,
                   help="Within-family percentile below which a chunk is negative.")
    p.add_argument("--null_spread", type=float, default=0.02,
                   help="A family whose advantage IQR is below this is labelled null "
                        "wholesale -- there is nothing to prefer, and thresholding would "
                        "manufacture a distinction the data does not contain.")
    p.add_argument("--exclude_tile_holdout", action="store_true",
                   help="Drop windows on held-out tiles from the label (they are the "
                        "generalisation measure; training on them spends it).")
    args = p.parse_args()

    d = np.load(args.predictions)
    key = f"p_{args.variant}"
    if key not in d:
        raise SystemExit(f"{args.predictions} has no {key}; it holds "
                         f"{[k for k in d.files if k.startswith('p_')]}")
    folds = int(d["num_folds"]) if "num_folds" in d.files else 1
    if folds == 1:
        print("[label] WARNING: the critic was fit with --folds 1, so most of these "
              "predictions are IN SAMPLE -- the critic has seen the outcome of the very "
              "episode it is scoring. Under memorisation V(s_t) and V(s_{t+n}) both collapse "
              "onto the realized outcome and the advantage goes to zero, so the label degrades "
              "to noise rather than becoming wrong. Refit with --folds 2 (same wall-clock).")

    p_fail = d[key]
    uid, ridx, fam = d["episode_uid"], d["replan_index"], d["family_code"]
    failed, val_tile = d["failed"].astype(bool), d["val_tile"].astype(bool)

    a, boot = adv_mod.n_step_advantage(p_fail, uid, ridx, failed, args.n)
    labels, per_family = adv_mod.binarize(
        a, fam, percentile=args.percentile, null_spread=args.null_spread)
    if args.exclude_tile_holdout:
        labels = np.where(val_tile, adv_mod.LABEL_NULL, labels)

    counts = {name: int((labels == i).sum()) for i, name in enumerate(adv_mod.LABEL_NAMES)}
    print(f"[label] n={args.n} re-plans, variant {args.variant}, {folds} fold(s)")
    print(f"[label] advantage: mean {a.mean():+.4f}  sd {a.std():.4f}  "
          f"|A|>0.05 on {float((np.abs(a) > 0.05).mean()):.1%} of windows  "
          f"({float(boot.mean()):.1%} bootstrapped on the outcome)")
    print(f"[label] labels: {counts}")
    for f, r in sorted(per_family.items()):
        tag = "NULL (no spread)" if r["null"] else \
            f"thr {r['threshold']:+.4f}  positive {r['positive_frac']:.1%}"
        print(f"    family {f}: n={r['n']:7d}  IQR {r['iqr']:.4f}  {tag}")

    gate = _bandwise_gate(p_fail, labels, failed)
    print("[label] G2 -- realized failure rate by label, WITHIN P(fail|s_t) bands:")
    print("        (the unbanded comparison is true by construction; only this one is a test)")
    for r in gate:
        if r["delta"] is None:
            print(f"    [{r['p_lo']:.3f}, {r['p_hi']:.3f})  n_pos {r['n_pos']:6d} "
                  f"n_neg {r['n_neg']:6d}   -- too few to report")
        else:
            print(f"    [{r['p_lo']:.3f}, {r['p_hi']:.3f})  n_pos {r['n_pos']:6d} "
                  f"n_neg {r['n_neg']:6d}   fail pos {r['fail_pos']:.3f} / "
                  f"neg {r['fail_neg']:.3f}   delta {r['delta']:+.3f}")
    good = [r["delta"] for r in gate if r["delta"] is not None]
    if good:
        print(f"[label] G2 median band delta {float(np.median(good)):+.3f} "
              f"({sum(x > 0 for x in good)}/{len(good)} bands positive). A delta at or below "
              "zero means the label is reproducing state difficulty and conditioning on it "
              "cannot work.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    meta = {"n": int(args.n), "variant": args.variant, "percentile": float(args.percentile),
            "null_spread": float(args.null_spread), "num_folds": folds,
            "predictions": os.path.abspath(args.predictions),
            "counts": counts, "per_family": {str(k): v for k, v in per_family.items()},
            "g2_bands": gate}
    np.savez(args.out, label=labels.astype(np.int8), advantage=a.astype(np.float32),
             bootstrapped=boot, episode_uid=uid, replan_index=ridx, family_code=fam,
             val_tile=val_tile, meta=json.dumps(meta))
    print(f"[label] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
