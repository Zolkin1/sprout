# Copyright 2026 Zachary Olkin. All rights reserved.

"""Separate MODE COMMITMENT from a weak conditioning signal, on one planner checkpoint.

The planner decodes ``N`` tokens autoregressively. Position 0 is measured to be both the
most uncertain position and the least accurate one, and every later token is conditioned on
it -- so a wrong first token yields a coherent continuation of the WRONG mode (a stairs plan
on flat ground) rather than a slightly-off version of the right one. That is a different
failure from the pooled RMSE, which averages it away.

Four arms over one val split, differing ONLY in which tokens are free:

``oracle_all``  every token forced to the target. A CONTROL: it decodes the target ids
                through the frozen decoder, so it must reproduce the tokenizer's own
                reconstruction error. If it does not, this harness is wrong, not the model.
``oracle_t0``  position 0 forced, 1..N-1 greedy. The measurement.
``greedy``     nothing forced, argmax -- what Isaac serves.
``sampled``    nothing forced, one categorical draw at ``--temperature``.

Read ``oracle_t0`` against ``greedy``: the gap that closes is what first-token commitment
costs, and the gap that REMAINS above ``oracle_all`` is what the conditioning cannot resolve
no matter how the tokens are drawn.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/ablate_first_token.py \\
        --planner logs/token_planner/<run>/ckpt_40.pt \\
        --fm_config logs/flow_matching/<fm run>/params/fm_config.yaml \\
        --data <recorded>.h5,<augmented>.h5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_gen.features import _REPO_ROOT

from robot_rl.flow_matching.dataset import build_train_val
from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig
from robot_rl.token_planner.data import load_recordings
from robot_rl.token_planner.metrics import (
    ReconAccumulator,
    channel_group_labels,
    format_report,
)
from robot_rl.token_planner.planner import TokenPlanner
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer


def _load_trainer():
    """Import ``train_planner`` as a module, to reuse its dataset wrapper verbatim.

    Returns:
        The imported module.
    """
    path = _REPO_ROOT / "scripts" / "token_planner" / "train_planner.py"
    spec = importlib.util.spec_from_file_location("train_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_planner"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    """CLI.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planner", required=True, help="Planner checkpoint (.pt).")
    p.add_argument("--fm_config", required=True, help="The generator's saved fm_config.yaml.")
    p.add_argument("--data", required=True, help="Comma-separated recording paths.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--raw_weights", action="store_true",
                   help="Use `model` rather than the preferred `model_ema`.")
    p.add_argument("--out", default=None, help="Write the full report dicts here as JSON.")
    return p.parse_args()


def main() -> None:
    """Run the four arms and print one table."""
    args = parse_args()
    tp = _load_trainer()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ck = torch.load(args.planner, map_location="cpu", weights_only=False)
    cfg = PlannerConfig.from_dict(ck["config"])
    tok_cfg = TokenizerConfig.from_dict(cfg.tokenizer)
    model = TokenPlanner(cfg, TrajectoryTokenizer(tok_cfg))
    state = ck["model"] if args.raw_weights else (ck.get("model_ema") or ck["model"])
    model.load_state_dict(state)
    model = model.to(device).eval().requires_grad_(False)
    print(f"[abl] planner epoch {ck.get('epoch')} "
          f"weights={'raw' if args.raw_weights else 'EMA'} N={model.num_tokens} "
          f"V={model.vocab_size}")

    fm = tp.load_fm_config(args.fm_config)
    from robot_rl.token_planner.data import _EXTERO_BY_COST, _recorded_data_io
    keys = [k for k in _recorded_data_io().WINDOW_ARRAY_KEYS
            if k not in set(_EXTERO_BY_COST) - {fm.extero_source}]
    data, meta = load_recordings(args.data.split(","), keys=keys)
    train_ds, val_ds = build_train_val(data, meta, fm)
    if val_ds is None or not len(val_ds):
        raise SystemExit("no validation split")

    fam_names = [str(s) for s in (getattr(meta, "family_names", None) or ["unknown"])]
    fam_all = data.get("clip_family")
    sel = getattr(val_ds, "window_sel", None)
    fam = (torch.as_tensor(np.asarray(fam_all)[np.asarray(sel)], dtype=torch.long)
           if fam_all is not None and sel is not None
           else torch.zeros(len(val_ds), dtype=torch.long))

    tokens = tp.tokenize_split(model.tokenizer, val_ds.state, device)
    loader = DataLoader(tp.TupleDataset(val_ds, tokens, fam), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    group_names, labels = channel_group_labels(
        int(val_ds.num_joints), int(val_ds.num_extra_frames), tok_cfg.reference_blocks)
    norm = val_ds.state_normalizer

    def fresh() -> ReconAccumulator:
        """A trajectory-error accumulator on this run's layout."""
        return ReconAccumulator(
            group_names=group_names, labels=labels, std=norm.std.clone(),
            mean=norm.mean.clone(), chunk_len=tok_cfg.chunk_len,
            vocab_size=tok_cfg.vocab_size, num_families=len(fam_names),
            family_names=fam_names)

    arms = ["oracle_all", "oracle_t0", "greedy", "sampled"]
    acc = {a: fresh() for a in arms}
    gen = torch.Generator(device=device.type).manual_seed(args.seed)
    std_dev = norm.std.to(device)
    n_t0_hit = {a: 0 for a in arms}
    n_win = 0
    # base_pos squared error split by whether the FREE arm's token 0 matched the target.
    sq_hit, n_hit, sq_miss, n_miss = 0.0, 0, 0.0, 0
    bp = group_names.index("base_pos") if "base_pos" in group_names else 0

    for batch in loader:
        hist, extero, cmd, _ = tp._cond(batch, device)
        tgt_ids = batch["tokens"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        family = batch["family"]
        n_win += tgt_ids.shape[0]
        free = torch.full_like(tgt_ids, -1)
        t0_only = free.clone()
        t0_only[:, 0] = tgt_ids[:, 0]
        cache = model.encode_prefix(hist, extero, cmd, None)
        for arm in arms:
            force = {"oracle_all": tgt_ids, "oracle_t0": t0_only}.get(arm)
            ids, _, _ = model.sample(
                hist, extero, cmd, None, temperature=args.temperature,
                greedy=(arm != "sampled"),
                generator=(None if arm != "sampled" else gen),
                prefix_cache=cache, force_tokens=force)
            # `decode` with no normalizer, exactly as the trainer's `evaluate` does: the
            # nodes stay NORMALIZED and `ReconAccumulator` owns the conversion to physical
            # units, so these numbers are comparable with the training curve.
            pred = model.decode(ids)
            acc[arm].update(state, pred, ids, family)
            n_t0_hit[arm] += int((ids[:, 0] == tgt_ids[:, 0]).sum())
            if arm == "greedy":
                hit = ids[:, 0] == tgt_ids[:, 0]
                err = (pred - state) * std_dev
                # per-window mean squared error over the base_pos channels
                mask = torch.as_tensor(labels, device=err.device) == bp
                e = (err[..., mask] ** 2).mean(dim=(1, 2))
                sq_hit += float(e[hit].sum()); n_hit += int(hit.sum())
                sq_miss += float(e[~hit].sum()); n_miss += int((~hit).sum())

    reports = {a: acc[a].report() for a in arms}
    ks = [k for k in ("base_pos", "base_ori", "base_lin_vel", "base_ang_vel", "joint_pos")
          if k in reports["greedy"]["rmse"]]
    print(f"\nval windows: {n_win}\n")
    print(f"{'arm':<14}{'tok0 acc':>10}" + "".join(f"{k:>15}" for k in ks))
    for a in arms:
        r = reports[a]["rmse"]
        print(f"{a:<14}{n_t0_hit[a] / max(n_win, 1):>10.3f}"
              + "".join(f"{r.get(k, float('nan')):>15.5f}" for k in ks))
    print(f"\n{'arm':<14}{'':>10}" + "".join(f"{k:>15}" for k in ks) + "   (node p99)")
    for a in arms:
        r = reports[a]["node_rmse_p99"]
        print(f"{a:<14}{'':>10}" + "".join(f"{r.get(k, float('nan')):>15.5f}" for k in ks))
    if n_hit and n_miss:
        print(f"\ngreedy base_pos RMSE | token0 CORRECT ({n_hit}): "
              f"{(sq_hit / n_hit) ** 0.5:.5f} m")
        print(f"greedy base_pos RMSE | token0 WRONG   ({n_miss}): "
              f"{(sq_miss / n_miss) ** 0.5:.5f} m")
    print("\n--- per family, base_pos [m] ---")
    print(f"{'family':<28}" + "".join(f"{a:>13}" for a in arms))
    for f in sorted(reports["greedy"]["rmse_by_family"]):
        print(f"{f:<28}" + "".join(
            f"{reports[a]['rmse_by_family'].get(f, {}).get('base_pos', float('nan')):>13.5f}"
            for a in arms))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"n_windows": n_win, "tok0_acc": {a: n_t0_hit[a] / n_win for a in arms},
                       "reports": reports}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
