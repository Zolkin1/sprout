# Copyright 2026 Zachary Olkin. All rights reserved.

"""Does the advantage label move the served window by more than ordinary sampling noise?

The cheap screen that decides whether a full Isaac eval is worth an hour of GPU.  It answers
the one question ``flow_mse`` cannot: the training loss falls whether or not the label is
doing anything, because more gradient steps on rollout data fit that data better either way.

Three quantities, and the RATIO is the result:

* **label effect** -- ``positive`` vs ``negative`` at IDENTICAL conditioning and an identical
  noise draw, so the label is the only difference.
* **natural sampling noise** -- the pretrained model sampled twice from the same conditioning
  with two noise draws.  This is the floor the label has to clear to matter at all: the
  deployed sampler already lives with this much variation, and ``fm_variance_probe`` measured
  it flipping 20-35% of outcomes on real terrain.
* **model drift** -- pretrained vs fine-tuned at the same noise and label, i.e. how far the
  fine-tune moved the model REGARDLESS of the label.  Distinguishes "nothing trained" from
  "something trained but not through the label", which the first two cannot separate.

VALIDATED against the expensive instrument on three checkpoints -- 0.003, 0.006 and 0.007
all produced Isaac evals where the negative control matched or beat the positive arm:

    time_embed, 1x emb lr       0.003    eval: ac_pos 0.621 vs ac_neg 0.617   (null)
    time_embed, 10x model lr    0.006    eval: pos   0.622 vs neg   0.624   (null)
    time_embed, 50x emb lr      0.007    not run -- predicted null, between two measured ones

So a ratio in that range is a null before the eval is run.  It is a SCREEN and not a verdict:
it measures one window in isolation, and small per-window differences could in principle
compound over the ~25 re-plans of an episode in a way this cannot see.  A high ratio earns an
eval; a low one predicts a null but does not strictly prove it.

Usage::

    python scripts/flow_matching/label_divergence.py <ckpt-dir> --pretrained <base.pt> \\
        --rollout <rollout.h5>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "source", "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_rl.flow_matching import online  # noqa: E402


def _stats(name: str, a: torch.Tensor, b: torch.Tensor) -> float:
    """Print and return the mean absolute difference between two served window batches."""
    d = (a - b).abs()
    print(f"  {name:38s} mean {d.mean():.4f}  median {d.median():.4f}  max {d.max():.4f}")
    return float(d.mean())


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_dir", help="directory holding best_positive.pt / best_negative.pt")
    p.add_argument("--pretrained", required=True, help="the base checkpoint it came from")
    p.add_argument("--rollout", required=True, help="rollout .h5 to draw conditioning from")
    p.add_argument("--n", type=int, default=256, help="windows to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import h5py
    with h5py.File(args.rollout, "r") as f:
        n_win = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_win, size=min(args.n, n_win), replace=False))
        hist = torch.as_tensor(f["history"][idx].astype(np.float32))
        ext = torch.as_tensor(f["extero"][idx].astype(np.float32))
        cmd = torch.as_tensor(f["cmd_future"][idx].astype(np.float32))
        nullh = torch.as_tensor(f["history_null"][idx].astype(bool))

    dev = args.device
    pos = online.load_fm_context(os.path.join(args.ckpt_dir, "best_positive.pt"), device=dev)
    neg = online.load_fm_context(os.path.join(args.ckpt_dir, "best_negative.pt"), device=dev)
    base = online.load_fm_context(args.pretrained, device=dev)

    def build(ctx):
        """The three normalized token streams this context expects."""
        return (online.build_history_tokens(ctx, hist.to(dev)),
                online.build_extero_tokens(ctx, ext.to(dev)),
                online.build_cmd_tokens(ctx, cmd.to(dev)))

    nm = nullh.to(dev)
    n = idx.size
    noise_a = torch.randn(n, pos.F, pos.D,
                          generator=torch.Generator().manual_seed(args.seed)).to(dev)
    noise_b = torch.randn(n, pos.F, pos.D,
                          generator=torch.Generator().manual_seed(args.seed + 1)).to(dev)

    served_pos = online.sample_window(pos, *build(pos), history_null_mask=nm, noise=noise_a)
    served_neg = online.sample_window(neg, *build(neg), history_null_mask=nm, noise=noise_a)
    base_a = online.sample_window(base, *build(base), history_null_mask=nm, noise=noise_a)
    base_b = online.sample_window(base, *build(base), history_null_mask=nm, noise=noise_b)

    print(f"\nn={n} windows from {os.path.basename(args.rollout)}\n")
    print("LABEL EFFECT (positive vs negative, SAME noise):")
    d_label = _stats("fine-tuned pos vs neg", served_pos, served_neg)
    print("\nNATURAL SAMPLING NOISE (pretrained, two noise draws):")
    d_noise = _stats("pretrained noise_a vs noise_b", base_a, base_b)
    print("\nMODEL DRIFT (pretrained vs fine-tuned, same noise and label):")
    d_drift = _stats("base vs fine-tuned positive", base_a, served_pos)

    print(f"\nratio label/noise: {d_label / d_noise:.3f}   <-- the screen")
    print(f"ratio drift/noise: {d_drift / d_noise:.3f}   <-- did the model move AT ALL")
    print("\n  measured nulls for reference: 0.003 / 0.006 / 0.007 (time_embed injection)")
    if d_label / d_noise < 0.02:
        print("  READ: at or near the range that produced flat evals. A full eval is very "
              "likely to show the negative control matching the positive arm.")
    else:
        print("  READ: above the range that produced flat evals -- an eval can distinguish "
              "this, and is worth running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
