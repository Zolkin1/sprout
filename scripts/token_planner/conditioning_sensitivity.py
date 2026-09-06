# Copyright 2026 Zachary Olkin. All rights reserved.

"""Does the planner actually USE its conditioning? One number per stream.

Symptom this exists for: sampled plans wander across modes (a clean stairs chunk, then a
walk, then junk) from what is nearly the same observation -- which is what a model that has
learned the MARGINAL distribution over plans looks like, as opposed to a conditional one.

The measurement is an intervention, not a correlation. For each conditioning stream the
stream is replaced by ANOTHER window's (a roll along the batch, so the substitute is a real,
plausible observation rather than noise) and the teacher-forced cross-entropy is recomputed
on the SAME targets. The rise is how much predictive information that stream carries:

* ``dCE ~ 0``           the model ignores the stream. No decode rule recovers what the
                        network never conditioned on.
* ``dCE`` large          the stream is used, and a wandering SAMPLE is a decode/uncertainty
                        problem rather than a conditioning one.

``history_null`` is the learned null embedding rather than a roll -- the model is TRAINED on
it for ``null_cond_prob`` of samples, so it is the in-distribution way to say "no history"
and bounds what removing history can cost.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/conditioning_sensitivity.py \\
        --planner logs/token_planner/<run>/ckpt_40.pt \\
        --fm_config logs/flow_matching/<fm run>/params/fm_config.yaml \\
        --data <recorded>.h5 --load_rows 40000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from motion_gen.features import _REPO_ROOT

from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig
from robot_rl.token_planner.data import load_recordings
from robot_rl.token_planner.planner import TokenPlanner
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer


def _load_trainer():
    """Import ``train_planner`` to reuse its dataset wrapper and config loader.

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
    p.add_argument("--planner", required=True)
    p.add_argument("--fm_config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--load_rows", type=int, default=0,
                   help="Truncate the recording. Fine here: this measures a DIFFERENCE "
                        "between arms on identical windows, not an absolute quality.")
    p.add_argument("--max_batches", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--raw_weights", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run every intervention and print the table."""
    args = parse_args()
    tp = _load_trainer()
    device = torch.device(args.device)

    ck = torch.load(args.planner, map_location="cpu", weights_only=False)
    cfg = PlannerConfig.from_dict(ck["config"])
    tok_cfg = TokenizerConfig.from_dict(cfg.tokenizer)
    model = TokenPlanner(cfg, TrajectoryTokenizer(tok_cfg))
    model.load_state_dict(ck["model"] if args.raw_weights
                          else (ck.get("model_ema") or ck["model"]))
    model = model.to(device).eval().requires_grad_(False)
    print(f"[cond] planner epoch {ck.get('epoch')} N={model.num_tokens} "
          f"null_cond_prob={cfg.null_cond_prob}")

    fm = tp.load_fm_config(args.fm_config)
    from robot_rl.token_planner.data import _EXTERO_BY_COST, _recorded_data_io
    keys = [k for k in _recorded_data_io().WINDOW_ARRAY_KEYS
            if k not in set(_EXTERO_BY_COST) - {fm.extero_source}]
    if args.load_rows > 0:
        tp._patch_row_limit(args.load_rows)
    data, meta = load_recordings(args.data.split(","), keys=keys)
    train_ds, val_ds = build = tp.build_train_val(data, meta, fm)
    ds = val_ds if (val_ds is not None and len(val_ds)) else train_ds
    toks = tp.tokenize_split(model.tokenizer, ds.state, device)
    fam = torch.zeros(len(ds), dtype=torch.long)
    loader = DataLoader(tp.TupleDataset(ds, toks, fam), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    arms = ["intact", "extero_rolled", "history_rolled", "history_null",
            "cmd_rolled", "all_rolled"]
    tot = {a: 0.0 for a in arms}
    pos = {a: torch.zeros(model.num_tokens, dtype=torch.float64) for a in arms}
    n = 0
    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break
        hist, extero, cmd, _ = tp._cond(batch, device)
        tgt = batch["tokens"].to(device, non_blocking=True)
        b = tgt.shape[0]
        if b < 2:
            continue
        n += b
        for arm in arms:
            h, e, c = hist, extero, cmd
            null = None
            if arm in ("extero_rolled", "all_rolled"):
                e = torch.roll(extero, 1, dims=0)
            if arm in ("history_rolled", "all_rolled"):
                h = torch.roll(hist, 1, dims=0)
            if arm in ("cmd_rolled", "all_rolled") and cmd is not None:
                c = torch.roll(cmd, 1, dims=0)
            if arm == "history_null":
                null = torch.ones(b, dtype=torch.bool, device=device)
            logits = model(tgt, h, e, c, null)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 tgt.reshape(-1), reduction="none").view(b, -1)
            tot[arm] += float(ce.mean(dim=1).sum())
            pos[arm] += ce.sum(dim=0).double().cpu()

    base = tot["intact"] / n
    print(f"\nwindows: {n}   (teacher-forced CE, nats/token)\n")
    print(f"{'arm':<18}{'CE':>9}{'dCE vs intact':>16}")
    for a in arms:
        print(f"{a:<18}{tot[a] / n:>9.4f}{tot[a] / n - base:>16.4f}")
    print(f"\n{'per position':<18}" + "".join(f"{i:>7}" for i in range(model.num_tokens)))
    for a in arms:
        d = (pos[a] - pos["intact"]) / n
        print(f"{a:<18}" + "".join(f"{v:>7.3f}" for v in d))
    print("\n(rows are dCE per decode position; position 0 is where the mode is chosen)")


if __name__ == "__main__":
    main()
