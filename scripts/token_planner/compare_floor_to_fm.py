# Copyright 2026 Zachary Olkin. All rights reserved.

"""Score the FM and the tokenizer's floor with ONE metric on ONE set of windows.

The two numbers usually quoted for these come from different code paths and are not
comparable as written: the flow trainer's ``channel_metrics`` returns a PER-BATCH RMSE which
``train.py`` then averages over batches (a mean of RMSEs, which Jensen puts at or below a
pooled one), while ``ReconAccumulator`` reports a pooled ``sqrt(sum sq / sum n)``. They are
also different QUANTITIES -- the flow number is prediction error, the tokenizer's is
reconstruction of the true future -- and the flow one is a single Gaussian draw.

This runs three arms through the same accumulator on the same held-out windows:

``tokenizer``   ``decode(encode(future))`` -- the floor any tokenized planner is bounded by.
``fm_sample``   the flow generator's own draw, as ``train.py`` scores it.
``fm_zero``     the flow generator at ``x0 = 0``, the deterministic arm the controller runs.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/compare_floor_to_fm.py \\
        --tokenizer <tok>.pt --fm <fm run>/best.pt --fm_config <fm run>/params/fm_config.yaml \\
        --data <recorded>.h5,<augmented>.h5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import torch
from torch.utils.data import DataLoader

from motion_gen.features import _REPO_ROOT

from robot_rl.flow_matching.flow import sample_trajectory
from robot_rl.token_planner.config import TokenizerConfig
from robot_rl.token_planner.data import load_recordings
from robot_rl.token_planner.metrics import ReconAccumulator, channel_group_labels, format_report
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer


def _trainer():
    """Import ``train_planner`` for its dataset plumbing."""
    path = _REPO_ROOT / "scripts" / "token_planner" / "train_planner.py"
    spec = importlib.util.spec_from_file_location("train_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_planner"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--fm", required=True)
    p.add_argument("--fm_config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_windows", type=int, default=0, help="0 = the whole val split.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    """Score all three arms and print one table."""
    args = parse_args()
    tp = _trainer()
    dev = torch.device(args.device)

    tok_ck = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    tok_cfg = TokenizerConfig.from_dict(tok_ck["config"])
    tok = TrajectoryTokenizer(tok_cfg)
    tok.load_state_dict(tok_ck.get("model_ema") or tok_ck["model"])
    tok = tok.to(dev).eval().requires_grad_(False)

    fm_ck = torch.load(args.fm, map_location="cpu", weights_only=False)
    fm_cfg = tp.load_fm_config(args.fm_config)
    from robot_rl.flow_matching.model import FlowMatchingTransformer  # noqa: PLC0415
    fm = FlowMatchingTransformer.from_config(fm_cfg)
    fm.load_state_dict(fm_ck.get("model_ema") or fm_ck["model"])
    fm = fm.to(dev).eval().requires_grad_(False)

    from robot_rl.token_planner.data import _EXTERO_BY_COST, _recorded_data_io
    keys = [k for k in _recorded_data_io().WINDOW_ARRAY_KEYS
            if k not in set(_EXTERO_BY_COST) - {fm_cfg.extero_source}]
    data, meta = load_recordings(args.data.split(","), keys=keys)
    fm_cfg.future_len = tok_cfg.chunk_len
    fm_cfg.reference_blocks = list(tok_cfg.reference_blocks)
    train_ds, val_ds = tp.build_train_val(data, meta, fm_cfg)
    tp.assert_normalizers_agree(train_ds.state_normalizer, tok_ck["state_normalizer"])
    norm = val_ds.state_normalizer.to(dev)
    fam = torch.zeros(len(val_ds), dtype=torch.long)
    toks = tp.tokenize_split(tok, val_ds.state, dev)
    loader = DataLoader(tp.TupleDataset(val_ds, toks, fam), batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    groups, labels = channel_group_labels(int(val_ds.num_joints),
                                          int(val_ds.num_extra_frames),
                                          tok_cfg.reference_blocks)

    def fresh() -> ReconAccumulator:
        """One accumulator, so every arm is reduced identically."""
        return ReconAccumulator(group_names=groups, labels=labels, std=norm.std.clone(),
                                mean=norm.mean.clone(), chunk_len=tok_cfg.chunk_len,
                                vocab_size=tok_cfg.vocab_size, num_families=1,
                                family_names=["all"])

    arms = ("tokenizer", "fm_sample", "fm_zero")
    acc = {a: fresh() for a in arms}
    n = 0
    with torch.no_grad():
        for batch in loader:
            if args.max_windows and n >= args.max_windows:
                break
            h, e, c, _ = tp._cond(batch, dev, 0.0)
            state = batch["state"].to(dev, non_blocking=True)
            ids = batch["tokens"].to(dev, non_blocking=True)
            n += state.shape[0]
            preds = {
                "tokenizer": tok.decode_ids(ids),
                "fm_sample": sample_trajectory(fm, h, e, c, fm_cfg.future_len,
                                               fm_cfg.state_dim, fm_cfg.num_sampling_steps),
                "fm_zero": sample_trajectory(fm, h, e, c, fm_cfg.future_len,
                                             fm_cfg.state_dim, fm_cfg.num_sampling_steps,
                                             zero_noise=True),
            }
            fam_b = batch["family"]
            for a in arms:
                acc[a].update(state, preds[a], ids, fam_b)

    print(f"\nwindows: {n}   (one pooled RMSE, one val split, physical units)\n")
    ks = [k for k in ("base_pos", "base_ori", "base_lin_vel", "base_ang_vel", "joint_pos")]
    print(f"{'arm':<12}" + "".join(f"{k:>15}" for k in ks))
    for a in arms:
        r = acc[a].report()["rmse"]
        print(f"{a:<12}" + "".join(f"{r.get(k, float('nan')):>15.5f}" for k in ks))
    print("\n`tokenizer` is the FLOOR: no choice of tokens beats it. The fm arms are "
          "prediction error, so the comparison says whether a perfect tokenized planner "
          "could match the generator it replaces -- not that they measure the same thing.")


if __name__ == "__main__":
    main()
