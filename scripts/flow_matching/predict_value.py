# Copyright 2026 Zachary Olkin. All rights reserved.

"""Rebuild a critic's out-of-fold predictions from its saved fold checkpoints.

``fit_value.py`` writes ``predictions.npz`` only after EVERY requested variant has
finished, so a run that is half done -- or one interrupted, or one whose second variant is
not wanted -- leaves its completed critics on disk with nothing to read them with.  This
reads them.

The split is not re-derived here, it is RE-EXECUTED: ``_load``, ``_splits`` and the fold
assignment are imported from ``fit_value`` and run with the same seed, so the out-of-fold
property is reproduced rather than approximated.  Reconstructing it independently would be
the obvious way to get this subtly wrong -- a fold boundary off by one episode silently
turns some predictions in-sample, which is the exact failure the folds exist to prevent, and
the resulting array would look entirely normal.

Usage::

    python scripts/flow_matching/predict_value.py logs/value/v2 \\
        --stem logs/rollouts/v2/rollout_<...> \\
        --fm_checkpoint <pretrained.pt> --variant V_priv --out logs/value/v2_priv_preview
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "source", "motion_gen"),
           os.path.join(_ROOT, "scripts", "rsl_rl"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fit_value  # noqa: E402
from robot_rl.flow_matching import online, value  # noqa: E402


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fit_dir", help="directory holding <variant>.fold<k>.pt")
    p.add_argument("--stem", required=True, help="rollout stem the critic was fit on")
    p.add_argument("--fm_checkpoint", required=True)
    p.add_argument("--variant", default="V_priv", choices=("V_priv", "V_onboard"))
    p.add_argument("--out", required=True, help="where to write predictions.npz + gates.json")
    p.add_argument("--val_frac", type=float, default=0.25, help="must match the fit")
    p.add_argument("--seed", type=int, default=0, help="must match the fit")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    folds = sorted(f for f in os.listdir(args.fit_dir)
                   if f.startswith(f"{args.variant}.fold") and f.endswith(".pt"))
    if not folds:
        raise SystemExit(f"{args.fit_dir} holds no {args.variant}.fold*.pt")
    print(f"[pred] {len(folds)} fold checkpoint(s): {folds}")

    W, E, lab = fit_value._load(args.stem, 0)
    keep = lab["live"]
    if not keep.all():
        W = {k: v[keep] for k, v in W.items()}
        lab = {k: v[keep] for k, v in lab.items()}
    uids = W["episode_uid"]
    tiles = np.array([E["ep_terrain"].get(int(u), -1) for u in uids])
    val_ep, val_tile = fit_value._splits(uids, tiles, args.val_frac, args.seed)

    # The SAME assignment fit_value makes, executed the same way -- see the module docstring.
    n_folds = len(folds)
    pool = np.unique(uids[~val_tile])
    rng_f = np.random.default_rng(args.seed + 1)
    assign = {int(u): int(i % n_folds) for i, u in enumerate(rng_f.permutation(pool))}
    fold_of = np.array([assign.get(int(u), -1) for u in uids])

    scan_hw = E["attrs"]["height_scan_hw"]
    ctx = online.load_fm_context(args.fm_checkpoint, device=args.device)
    priv = args.variant == "V_priv"
    p_all = np.zeros(uids.size, dtype=np.float64)
    for k, name in enumerate(folds):
        ck = torch.load(os.path.join(args.fit_dir, name), map_location="cpu",
                        weights_only=False)
        model = value.FlowValueModel(
            copy.deepcopy(ctx.model),
            scan_hw=(tuple(ck["scan_hw"]) if priv else None),
            proprio_dim=(int(ck["proprio_hw"][-1]) if priv else 0),
            proprio_steps=(int(ck["proprio_hw"][0]) if priv else 0),
            context_dim=int(ck["context_dim"]),
        ).to(args.device)
        model.load_state_dict(ck["model"])
        model.eval()
        va_k = (fold_of == k)
        idx = np.flatnonzero(va_k | val_tile)
        print(f"[pred] {name}: scoring {idx.size} windows "
              f"({int(va_k.sum())} own fold + {int(val_tile.sum())} held-out tiles)")
        pk = np.zeros(uids.size, dtype=np.float64)
        pk[idx] = fit_value._predict(model, ctx, W, idx, priv, args.device, scan_hw,
                                     args.batch)
        p_all[va_k] = pk[va_k]
        p_all[val_tile] += pk[val_tile] / n_folds
        del model
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    os.makedirs(args.out, exist_ok=True)
    results = {args.variant: {}}
    seen = (~val_tile) if n_folds > 1 else val_ep
    for scope, mask in (("val_seen_tiles", seen), ("val_UNSEEN_tiles", val_tile)):
        print(f"  -- {scope} --")
        results[args.variant][scope] = fit_value._gates(p_all, lab, W, mask)
    a_seen = results[args.variant]["val_seen_tiles"]["auc"]
    a_new = results[args.variant]["val_UNSEEN_tiles"]["auc"]
    print(f"[pred] generalisation gap (seen - unseen tiles): {a_seen - a_new:+.3f}")

    with open(os.path.join(args.out, "gates.json"), "w") as f:
        json.dump({"gap_mean_abs": float("nan"), "n_episodes": int(np.unique(uids).size),
                   "n_windows": int(uids.size), "results": results},
                  f, indent=2, default=fit_value._jsonable)
    np.savez(os.path.join(args.out, "predictions.npz"), episode_uid=uids,
             replan_index=W["replan_index"], val=val_ep, val_tile=val_tile, tile=tiles,
             failed=lab["failed"], first_fall_step=lab["first_fall_step"],
             family_code=lab["family_code"], num_folds=n_folds,
             **{f"p_{args.variant}": p_all})
    print(f"[pred] wrote {args.out}/predictions.npz and gates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
