# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compare generators on TRANSITION vs NON-TRANSITION val windows.

The pooled ``val_flow_mse`` cannot settle whether transition oversampling
(``--transition_fraction``) worked. Val deliberately keeps the NATURAL transition rate
(17.8% on the v5 library), so it is dominated by the 82% of windows that cross nothing,
and a model trained on a 50/50 mix looks slightly worse on it *by construction* -- even if
it is strictly better at the thing it was reweighted for. The trade-off the knob makes is
exactly what an average over both strata hides.

So: split the SAME val windows by whether they cross a terrain transition and report each
stratum separately. "Crosses a transition" is ground truth -- an ``anchor_mask_t`` boundary
inside ``[clip_step, clip_step + F)``, joined via the recorded ``clip_id``/``clip_step`` --
not a height heuristic.

Pass several checkpoints to get a paired table. They must share the recording, which also
makes the split identical (it is seeded from ``cfg.seed`` over episode ids), so the
comparison is per-window paired rather than two independent samples::

    python scripts/flow_matching/fm_error_by_transition.py \\
        --data logs/rsl_rl/g1/<run>/recorded_data/<rec>.h5 \\
        --checkpoint logs/flow_matching/<baseline>/best.pt \\
        --checkpoint logs/flow_matching/<treatment>/best.pt

Reports RMSE per reference block. A result is only interesting if the two strata MOVE IN
OPPOSITE DIRECTIONS: transition error down, non-transition error up, is the knob working.
Both up means it cost you nothing but bought nothing either.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "source", "robot_rl"))
sys.path.insert(0, os.path.join(_REPO, "source", "motion_gen"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))

from recorded_data_io import load_recorded_windows  # noqa: E402
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.dataset import (  # noqa: E402
    build_train_val,
    transition_window_mask,
)
from robot_rl.flow_matching.flow import sample_trajectory  # noqa: E402
from robot_rl.flow_matching.online import load_fm_context  # noqa: E402


def block_rmse(
    pred: torch.Tensor, target: torch.Tensor, block_slices: dict[str, slice]
) -> dict[str, np.ndarray]:
    """Per-window RMSE for each reference block.

    Args:
        pred: ``(W, F, D)`` denormalized prediction.
        target: ``(W, F, D)`` denormalized ground truth.
        block_slices: Block name -> column slice into ``D``.

    Returns:
        Block name -> ``(W,)`` per-window RMSE, so results can be stratified afterwards
        rather than pooled here.
    """
    out: dict[str, np.ndarray] = {}
    for name, slc in block_slices.items():
        e = (pred[..., slc] - target[..., slc]) ** 2       # (W, F, C)
        out[name] = e.mean(dim=(1, 2)).sqrt().cpu().numpy()
    out["all"] = ((pred - target) ** 2).mean(dim=(1, 2)).sqrt().cpu().numpy()
    return out


@torch.no_grad()
def per_window_errors(
    checkpoint: str, data: dict, meta, device: str, batch_size: int, max_windows: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, FlowMatchingConfig]:
    """Sample one checkpoint over its val split; return per-window RMSE + the val selector.

    Args:
        checkpoint: FM ``.pt`` (EMA weights are used by ``load_fm_context``).
        data: Recorded window arrays.
        meta: Recording metadata.
        device: Torch device string.
        batch_size: Eval batch size.
        max_windows: Cap on val windows scored (0 = all).

    Returns:
        ``(errors, val_sel, cfg)`` -- block name -> ``(Wv,)`` RMSE, the recorded-window
        indices those rows correspond to, and the checkpoint's config.

    Raises:
        SystemExit: If the checkpoint's data yields no val split.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    # The sampler knob must not perturb the eval split; only training used it.
    cfg.transition_fraction = None
    ctx = load_fm_context(checkpoint, device)
    _, val = build_train_val(data, meta, cfg)
    if val is None or len(val) == 0:
        raise SystemExit(f"{checkpoint}: no val split.")
    sel = getattr(val, "window_sel", None)
    if sel is None:
        raise SystemExit("val dataset has no window_sel; this needs the windowed .h5 path.")

    loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    acc: dict[str, list[np.ndarray]] = {}
    seen = 0
    for batch in loader:
        pred = sample_trajectory(
            ctx.model, batch["history"].to(device), batch["extero"].to(device),
            batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None,
            cfg.future_len, cfg.state_dim, cfg.num_sampling_steps,
        )
        p = ctx.state_norm.denormalize(pred)
        t = ctx.state_norm.denormalize(batch["state"].to(device))
        for k, v in block_rmse(p, t, val.block_slices).items():
            acc.setdefault(k, []).append(v)
        seen += p.shape[0]
        if max_windows and seen >= max_windows:
            break
    errors = {k: np.concatenate(v) for k, v in acc.items()}
    return errors, np.asarray(sel)[: len(next(iter(errors.values())))], cfg


def main() -> None:
    """Score every checkpoint on the shared val split and print the stratified table."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="The windowed .h5 the checkpoints trained on.")
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="FM .pt; repeat for a paired comparison (first = baseline).")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_windows", type=int, default=4000,
                    help="Cap val windows scored per checkpoint (0 = all).")
    ap.add_argument("--clip_library_path", type=str, default="",
                    help="Override the library to join clip_id against (default: recorded).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, meta = load_recorded_windows(args.data)
    W = int(data["history_state"].shape[0])

    results = []
    for ck in args.checkpoint:
        err, sel, cfg = per_window_errors(ck, data, meta, device, args.batch_size,
                                          args.max_windows)
        results.append((ck, err, sel, cfg))
        print(f"[eval] {os.path.basename(os.path.dirname(ck))}: {len(sel)} val windows scored")

    mcfg = FlowMatchingConfig.from_dict(
        torch.load(args.checkpoint[0], map_location="cpu", weights_only=False)["config"])
    if args.clip_library_path:
        mcfg.clip_library_path = args.clip_library_path
    trans_all = transition_window_mask(data, meta, mcfg, W)

    blocks = [k for k in results[0][1] if k != "all"] + ["all"]
    for block in blocks:
        print(f"\n=== {block} (RMSE) ===")
        print(f"  {'run':44s} {'transition':>12s} {'non-trans':>12s} {'pooled':>10s}")
        base = None
        for ck, err, sel, _ in results:
            m = trans_all[sel]
            e = err[block]
            row = (e[m].mean(), e[~m].mean(), e.mean())
            tag = "" if base is None else (
                f"   ({100 * (row[0] / base[0] - 1):+.1f}% / {100 * (row[1] / base[1] - 1):+.1f}%)")
            base = row if base is None else base
            # <run>/best.pt -> "<run>"; one dirname, not two (two lands on logs/flow_matching
            # and labels every row identically).
            name = os.path.basename(os.path.dirname(ck))[:44]
            print(f"  {name:44s} {row[0]:12.4f} {row[1]:12.4f} {row[2]:10.4f}{tag}")
        n_t = int(trans_all[results[0][2]].sum())
        print(f"  (val split: {n_t} transition / {len(results[0][2]) - n_t} non-transition)")


if __name__ == "__main__":
    main()
